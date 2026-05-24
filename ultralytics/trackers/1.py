class BYTETracker:
    """
    BYTETracker — 负责基于检测结果维护轨迹的主类（Kalman + IoU 关联 + 双阈值分步关联策略）

    主要数据结构（类成员）：
        tracked_stracks: 当前处于 `Tracked` 状态并持续被激活的轨迹列表（这些是正常输出的轨迹）
        lost_stracks: 暂时丢失但仍保留以等待短期内被重新识别的轨迹
        removed_stracks: 已经被彻底删除（或超过保留阈值）的轨迹（用于历史记录/剪裁）
        frame_id: 当前处理到的视频帧索引（从 1 开始）
        max_time_lost: 允许的最大丢失帧数（超过则从 lost -> removed）
        kalman_filter: 用来预测轨迹下一状态的 KalmanFilterXYAH 实例
        args: 参数组（见上文关键参数）
    """

    def __init__(self, args, frame_rate: int = 30):
        # 三个轨迹容器
        self.tracked_stracks = []  # type: list[STrack]   # 活跃并已激活的轨迹
        self.lost_stracks = []     # type: list[STrack]   # 暂时丢失（等待重识别）
        self.removed_stracks = []  # type: list[STrack]   # 被彻底移除的轨迹（历史）

        # 帧计数器
        self.frame_id = 0

        # 参数保存
        self.args = args

        # 根据视频帧率和 track_buffer 计算允许丢失的帧数：
        # 例如 frame_rate=30, track_buffer=30 -> max_time_lost = 30
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)

        # 每个 tracker 保留一个 KF 实例（可以复用，也可以为每个 STrack 创建独立 KF）
        self.kalman_filter = self.get_kalmanfilter()

        # STrack 使用的全局自增 id 需要在 tracker 初始化时重置
        self.reset_id()

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """
        主更新函数——对每一帧调用，输入检测结果并返回当前激活轨迹的列表（numpy array）。

        主要步骤（在代码中对应注释）：
            1. 增帧计数
            2. 根据置信度分割 detections（高置信 vs 低置信）
            3. 使用 init_track 将检测封装为 STrack 列表（每个 STrack 包含 bbox, score, cls, idx）
            4. 形成 strack_pool（tracked + lost），对其进行 KF 预测
            5. 可选：全局运动补偿（gmc）
            6. 第一轮关联（高置信）：线性分配（Hungarian）
            7. 第二轮关联（低置信）：对未匹配的仍然在跟踪的轨迹再和低置信检测尝试匹配
            8. 处理 unconfirmed（未确认）轨迹
            9. 初始化新轨迹（对仍未匹配且分数满足 new_track_thresh 的检测）
            10. 清理（转移 lost->removed，合并列表，去重）
            11. 返回当前激活轨迹结果数组
        """

        # 每调用一次 update 就认为处理下一帧
        self.frame_id += 1

        # 临时容器：用于收集这帧被激活/重识别/丢失/移除的轨迹（稍后统一合并）
        activated_stracks = []
        refind_stracks = []  # 重新找回（从lost回到tracked）
        lost_stracks = []
        removed_stracks = []

        # -------------------------
        # Step 1: split detections by score thresholds
        # -------------------------
        # results.conf 假设是 numpy array 或类似支持布尔索引的结构
        scores = results.conf

        # remain_inds: 用于第一轮关联（高置信）
        remain_inds = scores >= self.args.track_high_thresh

        # inds_low: 候选用于第二轮关联（score > low_thresh）
        inds_low = scores > self.args.track_low_thresh

        # inds_high: 区分出介于 low 和 high 之间的那部分（用于 second association）布尔类型掩码
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high  # 在 low 与 high 之间（低置信但不至于丢弃）
        results_second = results[inds_second]  # 低置信组
        results = results[remain_inds]         # 高置信组（用于第一轮）

        # feats_keep 与 feats_second：如果传入外观特征 feats，则按同样方式切分
        feats_keep = feats_second = img  # 默认为 img 占位（原代码很奇怪将 img 赋给 feats_keep）
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        # -------------------------
        # Step 2: wrap detections into STrack objects
        # -------------------------
        # convert detections into list[STrack]高置信度框
        detections = self.init_track(results, feats_keep)

        # tracked_stracks 包含已激活/未激活两类轨迹（we split into unconfirmed 和 tracked）
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]self.tracked_stracks state=tracked,有就获得和未激活的，没有lost
        for track in self.tracked_stracks:
            if not track.is_activated:
                # 尚未完全“激活”的轨迹（例如只在一帧出现过）
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)#存放第一类正常轨迹，激活且tracked

        # -------------------------
        # Step 3: First association （高置信组）
        # -------------------------
        # strack_pool = tracked + lost（我们允许lost也尝试与高置信检测匹配，这样可以找回）
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # 预测：对 strack_pool 的每条轨迹用 Kalman filter 预测下一位置
        self.multi_predict(strack_pool)

        # 可选：如果存在全局运动补偿模块（gmc），则把 strack_pool 和 unconfirmed 用相同变换 warp
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy)  # gmc.apply 根据帧间运动估计单应/仿射变换
            except Exception:
                warp = np.eye(2, 3)
            # 把变换应用到轨迹（坐标与协方差）
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 计算距离矩阵（默认 IoU distance，可选 fuse score）注意此时dectection是只有高置信度的框
        dists = self.get_dists(strack_pool, detections)

        # 匈牙利分配：得到匹配对 indices, 以及未匹配轨迹/检测索引
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 应用匹配结果：更新轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 常见路径：处于 tracked 的轨迹直接用检测 update
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # 其他情况：例如在 lost 中被找到 -> re_activate
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # -------------------------
        # Step 4: Second association（低置信组）——给未匹配轨迹另一次机会
        # -------------------------
        # 初始化低置信检测（同 init_track）
        detections_second = self.init_track(results_second, feats_second)

        # r_tracked_stracks：第一轮中没有被匹配到 but 本身 state==Tracked 的那些轨迹
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # 仅使用 IoU 距离去匹配这些轨迹与低置信检测（阈值通常更宽松）
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理第二轮仍未匹配到的轨迹：标记为 Lost（若不是 already Lost）
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # -------------------------
        # Step 5: 处理 unconfirmed（通常只出现一帧的短轨迹）
        # -------------------------
        # u_detection 对应第一轮没有被匹配的检测索引（高置信组中还未被用到的检测）
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 如果 unconfirmed 与某个检测匹配到 -> treat as activated（变成正式轨迹）
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # 未匹配到的 unconfirmed（通常是噪声/误检） -> 直接移除（mark_removed）
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # -------------------------
        # Step 6: 初始化新轨迹（对于高置信组中仍未匹配到的检测）
        # -------------------------
        for inew in u_detection:
            track = detections[inew]
            # 只有置信度足够高的新检测才会激活一个新轨迹，避免噪声太多
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # -------------------------
        # Step 7: 对 lost_stracks 做过期处理（超过 max_time_lost -> removed）
        # -------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # -------------------------
        # Step 8: 更新主列表（合并、去重、截断 removed 列表）
        # -------------------------
        # 只留下当前仍处于 TrackState.Tracked 的轨迹
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        # 把 activated、refind 合并进入 tracked 列表
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        # 更新 lost 列表：从 lost 中移除已经被找回或移走的
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 把本帧新加入的 lost 添加进 lost 列表
        self.lost_stracks.extend(lost_stracks)
        # 从 lost 中移除已经被标记 removed 的
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)

        # 去重：如果 tracked 与 lost 中出现了重复（IoU < 0.15 判为重复），保留较长时间的那一个
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 把本帧新标记为 removed 的加入 removed_stracks（历史）
        self.removed_stracks.extend(removed_stracks)
        # 为防止 removed 列表无限增长，保持最后 1000 个
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

        # 最后返回当前所有被激活的轨迹的结果数组（只包含 is_activated 的）
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """返回 KalmanFilterXYAH 实例（如果需要可以改为单例或配置化）"""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """
        将模型输出的 detection 集合转换为 STrack 实例列表（每个包含 bbox、score、cls、idx）。
        重要点：
            - results 预期含 `.xywh` 或 `.xywhr`（带旋转/宽高比等），选择合适的属性。
            - 在最后一列追加 detection 的 index（用于追踪的唯一标识/调试）
        """
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # 在 bbox 后面附加 detection index（用于 STrack.idx）
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        # 构造 STrack 列表：注意构造参数 (xywh_with_idx, score, cls)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """
        计算 tracks 与 detections 间的距离矩阵。
        默认使用 IoU 距离（matching.iou_distance），如果 args.fuse_score 为 True，还会调用 fuse_score
        （fuse_score 会把检测分数融合进距离矩阵，使得高分检测更容易被选中）
        """
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """对多个轨迹执行 Kalman 前向预测（封装到 STrack.multi_predict）"""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """重置 STrack 的自增 id 计数器（委托给 STrack）"""
        STrack.reset_id()

    def reset(self):
        """彻底重置 tracker 状态（清空所有列表，重置帧计数，重置 kalman 与 id）"""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """
        合并两个 STrack 列表，保持基于 track_id 的唯一性（先出现的保留）。
        用途：把 activated/refind 等并回 tracked 列表。
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """
        从 tlista 中去掉所有在 tlistb 中出现过的 track（以 track_id 作为判重依据）。
        用于：当一个 track 被找回或移除时，把它从 lost 列表中删除等场景。
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """
        在两个列表中检测并删除重复轨迹。重复判定基于 IoU 距离矩阵 pdist < 0.15。
        处理策略：
            - 对于检测到的重复对 (p,q)：
                比较两条轨迹的持续时间（frame_id - start_frame）：
                  - 持续时间更长的一方保留，短的一方删除
            - 返回去重后的 (resa, resb)
        备注：这是为了避免同一目标被两个轨迹重复跟踪（可能源自不同的 detection -> activation 时机）。
        """
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                # a 存活时间更久，删除 b 的副本
                dupb.append(q)
            else:
                # b 存活时间更久，删除 a 的副本
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb