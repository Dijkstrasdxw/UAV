"""
VisDrone MOT评估脚本 - 训练参数优化版
基于用户45% mAP的训练脚本参数进行优化

关键优化点：
1. 使用与训练相同的高分辨率设置 (imgsz=1280)
2. 采用训练时的小目标检测参数
3. 优化跟踪器配置以匹配检测性能
4. 支持GPU和CPU的自动调整

使用方式：
    python eval_visdrone_mot_optimized.py --model best.pt --root_dir VisDrone2019-MOT-val
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
import numpy as np
import torch

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    warnings.warn("Ultralytics not installed")

try:
    import motmetrics as mm
    HAS_MOTMETRICS = True
except ImportError:
    HAS_MOTMETRICS = False
    warnings.warn("py-motmetrics not installed. For comprehensive MOT metrics, install: pip install motmetrics")


class OptimizedVisDroneMOTEvaluator:
    """基于训练参数优化的评估器"""

    def __init__(self, model_path, root_dir, device='auto', tracker='bytetrack.yaml'):
        """初始化评估器

        Args:
            model_path (str): YOLO模型权重路径
            root_dir (str): 数据集根目录
            device (str): 'auto', 'cpu', 'cuda'
            tracker (str): 跟踪器配置
        """
        # 检查可用显存
        self.device = self._auto_select_device(device)
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)
            # 启用FP16加速（如果有足够显存）
            if torch.cuda.get_device_properties(0).total_memory > 16 * 1024**3:
                self.model.fuse()

        # ===== 关键修复：区分原始类别ID和YOLO类别ID =====
        
        # 原始VisDrone类别（GT标注中使用，0-11）
        self.original_class_names = {
            0: "ignored",           # 应过滤
            1: "pedestrian",        # MOT相关
            2: "people",            # MOT相关
            3: "bicycle",           # 过滤
            4: "car",               # MOT相关
            5: "van",               # MOT相关
            6: "truck",             # MOT相关
            7: "tricycle",          # 过滤
            8: "awning-tricycle",   # 过滤
            9: "bus",               # MOT相关
            10: "motor",            # 过滤
            11: "others"            # 应过滤
        }
        
        # YOLO训练输出类别（0-9，原始类别1-10映射）
        self.yolo_class_names = {
            0: "pedestrian",        # 原始1
            1: "people",            # 原始2
            2: "bicycle",           # 原始3
            3: "car",               # 原始4
            4: "van",               # 原始5
            5: "truck",             # 原始6
            6: "tricycle",          # 原始7
            7: "awning-tricycle",   # 原始8
            8: "bus",               # 原始9
            9: "motor"              # 原始10
        }
        
        # MOT相关类别：原始ID
        self.mot_original_classes = {1, 2, 4, 5, 6, 9}  # GT标注使用
        
        # ===== 关键修复：YOLO类别映射 =====
        # 如果YOLO是用yaml (0-9) 训练的，那么YOLO输出就是0-9
        # 这时候需要将YOLO的0-9映射回原始1-10来和GT比对
        # 但目前还不清楚YOLO实际输出什么，所以改为用类别名做过滤
        # 保留的MOT类别名
        self.mot_class_names = {"pedestrian", "people", "car", "van", "truck", "bus"}
        
        print(f"[INFO] 原始MOT类别: {self.mot_original_classes}")
        print(f"[INFO] MOT类别名: {self.mot_class_names}")

        # 使用与训练脚本相同的参数配置
        self.imgsz = 1280  # 与训练保持一致

        # 关键修复：降低置信度阈值到0.45，与ByteTrack的track_high_thresh一致
        self.conf_threshold = 0.45  # 降低到与ByteTrack一致
        self.iou_threshold = 0.7   # NMS的IoU阈值
        self.max_det = 200         # 限制检测数量

        print(f"[INFO] 推理参数 - imgsz: {self.imgsz}, conf: {self.conf_threshold}, iou: {self.iou_threshold}, max_det: {self.max_det}")

        # 设置跟踪器参数（基于训练优化）
        self.tracker = tracker
        self.tracker_config = self._get_optimized_tracker_config()

        # 数据集路径
        self.root_dir = Path(root_dir)
        self.sequences_dir = self.root_dir / 'sequences'
        self.annotations_dir = self.root_dir / 'annotations'
        self.results = defaultdict(lambda: {'metrics': {}})

    def _auto_select_device(self, device):
        """自动选择计算设备 - 强制使用GPU"""
        if device == 'auto':
            device = 'cuda'  # 强制使用GPU

        if device == 'cuda':
            if torch.cuda.is_available():
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
                return 'cuda'
            else:
                print("CUDA不可用，切换到CPU模式")
                return 'cpu'
        elif device == 'cpu':
            print("使用CPU模式")
            return 'cpu'
        else:
            print(f"未知设备: {device}, 使用CPU")
            return 'cpu'

    def _get_optimized_tracker_config(self):
        """获取优化的跟踪器配置"""
        config = {
            'bytetrack.yaml': {
                # 基于无人机场景调整跟踪参数
                'track_high_thresh': 0.45,    # 较高的跟踪阈值（减少ID切换）
                'track_low_thresh': 0.15,     # 较低的低阈值（保留更多轨迹）
                'new_track_thresh': 0.55,    # 新轨迹阈值（避免创建太多轨迹）
                'track_buffer': 45,           # 增加缓冲区长度（适应无人机运动）
                'match_thresh': 0.8,          # 关联阈值（保持稳定）
                'frame_rate': 30
            }
        }
        return config.get(self.tracker, config['bytetrack.yaml'])

    def track_sequence(self, seq_name):
        """逐帧跟踪单个序列（优化版）"""
        # 构建路径
        seq_dir = self.sequences_dir / seq_name
        gt_file = self.annotations_dir / f"{seq_name}.txt"

        if not seq_dir.exists():
            print(f"序列目录不存在: {seq_dir}")
            return None

        if not gt_file.exists():
            print(f"GT文件不存在: {gt_file}")
            return None

        # 加载GT数据
        gt_data = self._load_gt_file(str(gt_file))

        # 获取所有图像文件
        image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))

        if not image_files:
            print(f"序列中没有图像: {seq_dir}")
            return None

        # 使用生成器分批处理
        tracking_results = defaultdict(list)

        print(f"\n处理序列: {seq_name} ({len(image_files)} 帧)")
        print(f"优化模式: {self.device}, imgsz: {self.imgsz}")

        # 批处理（本地运行使用较小的batch size以节省内存）
        batch_size = 1  # 本地运行：batch_size=1，云GPU运行：batch_size=2
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc=f"批处理 {seq_name}"):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]

            # 批处理
            for frame_idx, img_file in enumerate(batch_files, batch_start + 1):
                try:
                    # 使用与训练相同的高分辨率推理
                    results = self.model.track(
                        str(img_file),
                        persist=True,
                        tracker=self.tracker,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_det,
                        imgsz=self.imgsz,
                        verbose=False,
                        device=self.device
                    )

                    # 提取跟踪结果
                    result = results[0]
                    if result.boxes is not None and hasattr(result.boxes, 'id'):
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # ��取track_id - 如果没有track_id则生成一个
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            else:
                                # 如果没有track_id，使用检测框的索引作为临时ID
                                track_id = hash(f"{frame_idx}_{int(box.cls[0])}_{int(box.conf[0]*1000)}") % 1000000

                            conf = float(box.conf[0])
                            cls = int(box.cls[0])

                            # ===== 关键修复：使用类别名而不是ID来过滤 =====
                            # 避免原始ID和YOLO ID的映射混淆
                            if cls < len(self.model.names):
                                cls_name = self.model.names[cls]
                            else:
                                cls_name = f"unknown_{cls}"
                            
                            if cls_name not in self.mot_class_names:
                                # 不在MOT类别中，过滤掉
                                continue

                            # 转换为 MOT 格式
                            w = x2 - x1
                            h = y2 - y1

                            tracking_results[frame_idx].append({
                                'frame_id': frame_idx,
                                'track_id': track_id,
                                'x': float(x1),
                                'y': float(y1),
                                'w': float(w),
                                'h': float(h),
                                'conf': float(conf),
                                'cls': int(cls)
                            })

                except Exception as e:
                    print(f"处理第 {frame_idx} 帧时出错: {e}")
                    continue

        # 使用 motmetrics 计算指标
        metrics = self._compute_mot_metrics_with_motmetrics(tracking_results, gt_data, seq_name)

        self.results[seq_name] = {
            'metrics': metrics,
            'num_frames': len(image_files),
            'device': self.device
        }

        return metrics

    def _load_gt_file(self, file_path):
        """加载VisDrone MOT格式的GT标注文件
        
        注意：GT文件中的类别ID是原始VisDrone格式（0-11）
        """
        data = defaultdict(list)

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(parts[7]) if len(parts) > 7 else 0
                    truncation = float(parts[8]) if len(parts) > 8 else 0.0
                    occlusion = float(parts[9]) if len(parts) > 9 else 0.0

                    # ===== 关键修复：使用原始类别ID过滤GT =====
                    # GT中的cls_id是原始VisDrone格式（0-11）
                    # 只保留MOT相关的原始类别 {1, 2, 4, 5, 6, 9}
                    if cls_id not in self.mot_original_classes:
                        # 不打印警告，这个过滤是正常的
                        continue

                    # ===== 关键修复：正确处理visibility =====
                    # VisDrone格式的occlusion范围是0-2（0=无遮挡, 1=部分遮挡, 2=严重遮挡）
                    # 转换为MOT格式的visibility（0-1）
                    # 无遮挡(0) -> 1.0, 部分遮挡(1) -> 0.5, 严重遮挡(2) -> 0.0
                    if occlusion == 0:
                        visibility = 1.0
                    elif occlusion == 1:
                        visibility = 0.5
                    else:  # occlusion == 2
                        visibility = 0.0

                    # ===== 过滤条件：只保留有效的框 =====
                    # 注意：occlusion=2（严重遮挡）的框不应保留（visibility=0）
                    if w > 0 and h > 0 and conf > 0 and visibility > 0:
                        data[frame_id].append({
                            'frame_id': frame_id,
                            'track_id': track_id,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'class': cls_id,
                            'visibility': visibility
                        })
        except Exception as e:
            print(f"❌ 读取GT文件失败: {e}")

        return data

    def _compute_mot_metrics_with_motmetrics(self, dt_results, gt_data, seq_name):
        """使用 motmetrics 库计算标准 MOT 指标"""
        if not HAS_MOTMETRICS:
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        print(f"使用 motmetrics 计算指标...")

        # 调试信息
        print(f"DT结果帧数: {len(dt_results)}")
        print(f"GT结果帧数: {len(gt_data)}")

        # 统计总数
        total_gt_boxes = sum(len(boxes) for boxes in gt_data.values())
        total_dt_boxes = sum(len(boxes) for boxes in dt_results.values())
        print(f"总GT框数: {total_gt_boxes}")
        print(f"总DT框数: {total_dt_boxes}")

        # 如果DT框数过多，显���警告
        if total_dt_boxes > total_gt_boxes * 10:
            print(f"警告: DT框数({total_dt_boxes})远大于GT框数({total_gt_boxes})")
            print("这可能表示类别过滤失效或置信度阈值过低")

        # 数据验证
        if not gt_data:
            print("警告: 没有GT数据")
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        if not dt_results:
            print("警告: 没有DT数据")
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        # 将结果转换为 motmetrics 格式
        # GT 格式: [frame_id, track_id, x, y, w, h, confidence, class_id, visibility]
        # DT 格式: [frame_id, track_id, x, y, w, h, confidence]

        gt_list = []
        for frame_id, boxes in gt_data.items():
            for box in boxes:
                gt_list.append([
                    frame_id,
                    box['track_id'],
                    box['x'],
                    box['y'],
                    box['w'],
                    box['h'],
                    1.0,  # GT confidence is always 1.0
                    1,    # class_id
                    box['visibility']  # ✅ 修正：使用实际计算的visibility
                ])

        dt_list = []
        for frame_id, boxes in dt_results.items():
            for box in boxes:
                dt_list.append([
                    frame_id,
                    box['track_id'],
                    box['x'],
                    box['y'],
                    box['w'],
                    box['h'],
                    box['conf']
                ])

        print(f"转换后GT数量: {len(gt_list)}")
        print(f"转换后DT数量: {len(dt_list)}")

        try:
            # 创建 accumulator - 使用正确的配置
            acc = mm.MOTAccumulator(auto_id=False)

            # 按 frame_id 组织数据
            frames = sorted(set([r[0] for r in gt_list] + [r[0] for r in dt_list]))
            print(f"处理帧数: {len(frames)}")

            mot_events = []

            for frame_id in frames:
                # 获取当前帧的GT
                gt_frame = [r for r in gt_list if r[0] == frame_id]
                # 获取当前帧的DT
                dt_frame = [r for r in dt_list if r[0] == frame_id]

                if gt_frame and dt_frame:
                    # 提取ID和坐标
                    gt_ids = [r[1] for r in gt_frame]
                    dt_ids = [r[1] for r in dt_frame]

                    # 计算距离矩阵 (使用1-IoU作为距离，这是标准的做法)
                    cost_matrix = np.full((len(gt_frame), len(dt_frame)), np.inf)
                    for i, gt_r in enumerate(gt_frame):
                        for j, dt_r in enumerate(dt_frame):
                            # 计算IoU
                            gt_bbox = [gt_r[2], gt_r[3], gt_r[2] + gt_r[4], gt_r[3] + gt_r[5]]
                            dt_bbox = [dt_r[2], dt_r[3], dt_r[2] + dt_r[4], dt_r[3] + dt_r[5]]
                            iou = self._bbox_iou(gt_bbox, dt_bbox)

                            # 使用标准的IoU距离计算：距离 = 1 - IoU
                            # 这样IoU越高，距离越小，符合motmetrics的要求
                            cost_matrix[i, j] = 1.0 - iou

                    # 更新accumulator
                    acc.update(gt_ids, dt_ids, cost_matrix, frameid=frame_id)

                elif gt_frame:
                    # 只有GT - 作为漏检
                    gt_ids = [r[1] for r in gt_frame]
                    acc.update(gt_ids, [], [], frameid=frame_id)

                elif dt_frame:
                    # 只有DT - 作为误检
                    dt_ids = [r[1] for r in dt_frame]
                    acc.update([], dt_ids, [], frameid=frame_id)

            # 计算 MOT 指标
            mh = mm.metrics.create()

            # 使用 motchallenge 指标集合
            metrics_to_compute = [
                'num_frames', 'num_objects', 'num_matches', 'num_false_positives',
                'num_misses', 'num_switches', 'mota', 'motp', 'idf1', 'precision',
                'recall', 'mostly_tracked', 'partially_tracked', 'mostly_lost'
            ]

            summary = mh.compute(acc, metrics=metrics_to_compute, name=seq_name)

            # 提取结果 - 使用更安全的方式
            if summary is not None and not summary.empty:
                # 获取第一行的结果
                row = summary.iloc[0]

                # 安全地获取指标值，使用float转换避免类型错误
                mota = float(row.get('mota', 0)) * 100
                motp = float(row.get('motp', 0))
                idf1 = float(row.get('idf1', 0)) * 100

                # 获取详细统计，使用int转换
                num_fps = int(row.get('num_false_positives', 0))
                num_misses = int(row.get('num_misses', 0))
                num_switches = int(row.get('num_switches', 0))
                num_frames = int(row.get('num_frames', 0))

            else:
                print("motmetrics 计算结果为空，使用手动计算")
                return self._compute_mot_metrics_manual(dt_results, gt_data)

            print(f"=== Motmetrics 结果 ===")
            print(f"MOTA: {mota:.2f}%")
            print(f"MOTP: {motp:.3f}")
            print(f"IDF1: {idf1:.2f}%")
            print(f"FP: {num_fps}, Misses: {num_misses}, Switches: {num_switches}")
            print(f"处理帧数: {num_frames}")

            # 计算TP - 使用预统计的总数
            tp = total_gt_boxes - num_misses  # TP = GT - Misses
            fn = num_misses
            fp = num_fps

            print(f"TP: {tp}, FP: {fp}, FN: {fn}")

            # 计算Precision和Recall
            if tp + fp > 0:
                precision = tp / (tp + fp) * 100
                print(f"Precision: {precision:.2f}%")
            else:
                precision = 0
                print("Precision: 0.00%")

            if tp + fn > 0:
                recall = tp / (tp + fn) * 100
                print(f"Recall: {recall:.2f}%")
            else:
                recall = 0
                print("Recall: 0.00%")

            # 检查MOTA值的合理性
            if mota < -100 or mota > 100:
                print("警告: MOTA值异常，可能存在数据格式问题")
                print(f"GT总数: {total_gt_boxes}, DT总数: {total_dt_boxes}")
                print(f"FP: {num_fps}, Misses: {num_misses}, Switches: {num_switches}")

            return {
                'MOTA': mota,
                'MOTP': motp,
                'IDF1': idf1,
                'num_frames': num_frames,
                'num_objects': total_gt_boxes,
                'num_false_positives': fp,
                'num_misses': fn,
                'num_switches': num_switches,
                'num_true_positives': tp,
                'precision': precision / 100 if precision > 0 else 0,
                'recall': recall / 100 if recall > 0 else 0
            }

        except Exception as e:
            print(f"motmetrics 计算失败: {e}")
            import traceback
            traceback.print_exc()
            print("使用手动计算的指标")
            return self._compute_mot_metrics_manual(dt_results, gt_data)

    def _compute_cost_matrix(self, gt_boxes, dt_boxes):
        """计算关联代价矩阵"""
        cost_matrix = np.zeros((len(gt_boxes), len(dt_boxes)))

        for i, gt_box in enumerate(gt_boxes):
            for j, dt_box in enumerate(dt_boxes):
                # 使用 IoU 作为相似度（距离 = 1 - IoU）
                iou = self._bbox_iou(
                    [gt_box['x'], gt_box['y'], gt_box['x'] + gt_box['w'], gt_box['y'] + gt_box['h']],
                    [dt_box['x'], dt_box['y'], dt_box['x'] + dt_box['w'], dt_box['y'] + dt_box['h']]
                )
                cost_matrix[i, j] = 1 - iou

        return cost_matrix

    def _compute_mot_metrics_manual(self, dt_results, gt_data):
        """手动计算 MOT 指标（备用版本）"""
        all_frames = sorted(set(list(dt_results.keys()) + list(gt_data.keys())))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # 调试：检查第一帧的匹配情况
        debug_frame = None
        for frame_id in all_frames:
            if frame_id in gt_data and frame_id in dt_results:
                debug_frame = frame_id
                break

        gt_boxes_list = []
        dt_boxes_list = []
        matches_list = []

        for frame_id in all_frames:
            gt_boxes = gt_data.get(frame_id, [])
            dt_boxes = dt_results.get(frame_id, [])

            total_gt += len(gt_boxes)

            # 计算匹配
            matches = self._match_boxes(gt_boxes, dt_boxes)

            if frame_id == debug_frame:
                gt_boxes_list = gt_boxes
                dt_boxes_list = dt_boxes
                matches_list = matches

            total_tp += len(matches)
            total_fp += len(dt_boxes) - len(matches)
            total_fn += len(gt_boxes) - len(matches)

        # 调试信息
        if debug_frame is not None:
            print(f"\n=== 调试信息 (第 {debug_frame} 帧) ===")
            print(f"GT框数: {len(gt_boxes_list)}, DT框数: {len(dt_boxes_list)}, 匹配数: {len(matches_list)}")
            print(f"前几个GT框: {gt_boxes_list[:2] if gt_boxes_list else '无'}")
            print(f"前几个DT框: {dt_boxes_list[:2] if dt_boxes_list else '无'}")

        # 简化的MOTA计算 - 限制在0-100%之间
        if total_gt > 0:
            mota = 1 - (total_fp + total_fn) / total_gt
            motp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

            # 限制MOTA在0-100%之间
            mota = max(0, min(1, mota))
        else:
            mota = 0
            motp = 0

        print(f"\n=== 详细指标计算 ===")
        print(f"总计 - GT: {total_gt}, TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"MOTA计算: 1 - ({total_fp} + {total_fn}) / {total_gt}")
        print(f"修正后的MOTA: {mota * 100:.2f}%")
        print(f"MOTP计算: {total_tp} / ({total_tp} + {total_fp}) = {motp:.3f}")

        return {
            'MOTA': mota * 100,
            'MOTP': motp,
            'IDF1': 0,  # 简化版本
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'ID_Switches': id_switches,
            'GT_Count': total_gt,
        }

    def _match_boxes(self, gt_boxes, dt_boxes):
        """匹配检测框和 GT"""
        matches = []
        for gt in gt_boxes:
            for dt in dt_boxes:
                iou = self._bbox_iou(
                    [gt['x'], gt['y'], gt['x'] + gt['w'], gt['y'] + gt['h']],
                    [dt['x'], dt['y'], dt['x'] + dt['w'], dt['y'] + dt['h']]
                )
                if iou > 0.5:
                    matches.append((gt, dt))
                    break
        return matches

    @staticmethod
    def _bbox_iou(box1, box2):
        """计算 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_all_sequences(self):
        """评估所有序列"""
        # 获取所有序列
        sequences = sorted([d.name for d in self.sequences_dir.iterdir() if d.is_dir()])

        all_metrics = {}

        print(f"\n找到 {len(sequences)} 个序列")
        print("="*70)

        for seq in sequences:
            try:
                metrics = self.track_sequence(seq)
                if metrics:
                    all_metrics[seq] = metrics
                    self._print_seq_metrics(seq, metrics)
            except Exception as e:
                print(f"处理序列 {seq} 时出错: {e}")

        # 汇总结果
        self._print_summary(all_metrics)

        return all_metrics

    def _print_seq_metrics(self, seq_name, metrics):
        """打印单个序列的指标"""
        print(f"\n{seq_name}:")
        print(f"   MOTA: {metrics.get('MOTA', 0):6.2f}%  |  MOTP: {metrics.get('MOTP', 0):6.3f}")
        if 'IDF1' in metrics:
            print(f"   IDF1: {metrics['IDF1']:6.2f}%")
        print(f"   TP: {metrics.get('TP', 0):5d}  FP: {metrics.get('FP', 0):5d}  FN: {metrics.get('FN', 0):5d}")

    def _print_summary(self, all_metrics):
        """打印汇总结果"""
        if not all_metrics:
            print("没有有效的评估结果")
            return

        # 计算平均值
        mota_scores = [m.get('MOTA', 0) for m in all_metrics.values()]
        motp_scores = [m.get('MOTP', 0) for m in all_metrics.values()]
        idf1_scores = [m.get('IDF1', 0) for m in all_metrics.values() if 'IDF1' in m]

        avg_mota = np.mean(mota_scores)
        avg_motp = np.mean(motp_scores)
        avg_idf1 = np.mean(idf1_scores) if idf1_scores else 0

        print("\n" + "="*70)
        print("VisDrone MOT 评估汇总（训练参数优化版）")
        print("="*70)

        print(f"\n整体平均指标 ({len(all_metrics)} 个序列):")
        print(f"  MOTA:      {avg_mota:6.2f}%")
        print(f"  MOTP:      {avg_motp:6.3f}")
        print(f"  IDF1:      {avg_idf1:6.2f}%")

        print(f"\n按序列详细结果:")
        print(f"{'序列名称':<30} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8}")
        print("-"*75)
        for seq_name in sorted(all_metrics.keys()):
            m = all_metrics[seq_name]
            print(f"{seq_name:<30} {m.get('MOTA', 0):>7.2f}% {m.get('MOTP', 0):>7.3f} {m.get('IDF1', 0):>7.2f}%")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VisDrone-MOT 评估（训练参数优化版）')
    parser.add_argument('--model', type=str, required=True, help='YOLO模型路径 (e.g., best.pt)')
    parser.add_argument('--root_dir', type=str, required=True, help='VisDrone2019-MOT数据集根目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='跟踪器配置')
    parser.add_argument('--seq', type=str, default=None, help='评估单个序列（可选）')
    parser.add_argument('--output', type=str, default='visdrone_mot_results.json', help='输出JSON文件')

    args = parser.parse_args()

    if not HAS_ULTRALYTICS:
        print("需要安装 ultralytics")
        print("   pip install ultralytics")
        return

    print("="*70)
    print("VisDrone MOT 评估工具（训练参数优化版）")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"数据: {args.root_dir}")
    print(f"跟踪器: {args.tracker}")
    print(f"设备: {args.device}")
    print("="*70)

    # 检查数据目录
    root_path = Path(args.root_dir)
    if not (root_path / 'sequences').exists() or not (root_path / 'annotations').exists():
        print("数据目录结构不正确，需要包含:")
        print("   ├── sequences/  (图像序列)")
        print("   └── annotations/  (GT标注)")
        return

    evaluator = OptimizedVisDroneMOTEvaluator(args.model, args.root_dir, args.device, args.tracker)

    if args.seq:
        # 评估单个序列
        print(f"\n评估单个序列: {args.seq}\n")
        metrics = evaluator.track_sequence(args.seq)
        if metrics:
            evaluator._print_seq_metrics(args.seq, metrics)
    else:
        # 评估所有序列
        print(f"\n评估所有序列\n")
        all_metrics = evaluator.evaluate_all_sequences()

        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {args.output}\n")


if __name__ == '__main__':
    main()