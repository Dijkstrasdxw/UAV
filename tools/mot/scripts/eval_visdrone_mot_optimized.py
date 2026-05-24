"""
VisDrone MOT璇勪及鑴氭湰 - 璁粌鍙傛暟浼樺寲鐗?
鍩轰簬鐢ㄦ埛45% mAP鐨勮缁冭剼鏈弬鏁拌繘琛屼紭鍖?

鍏抽敭浼樺寲鐐癸細
1. 浣跨敤涓庤缁冪浉鍚岀殑楂樺垎杈ㄧ巼璁剧疆 (imgsz=1280)
2. 閲囩敤璁粌鏃剁殑灏忕洰鏍囨娴嬪弬鏁?
3. 浼樺寲璺熻釜鍣ㄩ厤缃互鍖归厤妫€娴嬫€ц兘
4. 鏀寔GPU鍜孋PU鐨勮嚜鍔ㄨ皟鏁?

浣跨敤鏂瑰紡锛?
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
    """鍩轰簬璁粌鍙傛暟浼樺寲鐨勮瘎浼板櫒"""

    def __init__(self, model_path, root_dir, device='auto', tracker='bytetrack.yaml'):
        """鍒濆鍖栬瘎浼板櫒

        Args:
            model_path (str): YOLO妯″瀷鏉冮噸璺緞
            root_dir (str): 鏁版嵁闆嗘牴鐩綍
            device (str): 'auto', 'cpu', 'cuda'
            tracker (str): 璺熻釜鍣ㄩ厤缃?
        """
        # 妫€鏌ュ彲鐢ㄦ樉瀛?
        self.device = self._auto_select_device(device)
        print(f"浣跨敤璁惧: {self.device}")

        # 鍔犺浇妯″瀷
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)
            # 鍚敤FP16鍔犻€燂紙濡傛灉鏈夎冻澶熸樉瀛橈級
            if torch.cuda.get_device_properties(0).total_memory > 16 * 1024**3:
                self.model.fuse()

        # ===== 鍏抽敭淇锛氬尯鍒嗗師濮嬬被鍒獻D鍜孻OLO绫诲埆ID =====
        
        # 鍘熷VisDrone绫诲埆锛圙T鏍囨敞涓娇鐢紝0-11锛?
        self.original_class_names = {
            0: "ignored",           # 搴旇繃婊?
            1: "pedestrian",        # MOT鐩稿叧
            2: "people",            # MOT鐩稿叧
            3: "bicycle",           # 杩囨护
            4: "car",               # MOT鐩稿叧
            5: "van",               # MOT鐩稿叧
            6: "truck",             # MOT鐩稿叧
            7: "tricycle",          # 杩囨护
            8: "awning-tricycle",   # 杩囨护
            9: "bus",               # MOT鐩稿叧
            10: "motor",            # 杩囨护
            11: "others"            # 搴旇繃婊?
        }
        
        # YOLO璁粌杈撳嚭绫诲埆锛?-9锛屽師濮嬬被鍒?-10鏄犲皠锛?
        self.yolo_class_names = {
            0: "pedestrian",        # 鍘熷1
            1: "people",            # 鍘熷2
            2: "bicycle",           # 鍘熷3
            3: "car",               # 鍘熷4
            4: "van",               # 鍘熷5
            5: "truck",             # 鍘熷6
            6: "tricycle",          # 鍘熷7
            7: "awning-tricycle",   # 鍘熷8
            8: "bus",               # 鍘熷9
            9: "motor"              # 鍘熷10
        }
        
        # MOT鐩稿叧绫诲埆锛氬師濮婭D
        self.mot_original_classes = {1, 2, 4, 5, 6, 9}  # GT鏍囨敞浣跨敤
        
        # ===== 鍏抽敭淇锛歒OLO绫诲埆鏄犲皠 =====
        # 濡傛灉YOLO鏄敤yaml (0-9) 璁粌鐨勶紝閭ｄ箞YOLO杈撳嚭灏辨槸0-9
        # 杩欐椂鍊欓渶瑕佸皢YOLO鐨?-9鏄犲皠鍥炲師濮?-10鏉ュ拰GT姣斿
        # 浣嗙洰鍓嶈繕涓嶆竻妤歒OLO瀹為檯杈撳嚭浠€涔堬紝鎵€浠ユ敼涓虹敤绫诲埆鍚嶅仛杩囨护
        # 淇濈暀鐨凪OT绫诲埆鍚?
        self.mot_class_names = {"pedestrian", "people", "car", "van", "truck", "bus"}
        
        print(f"[INFO] 鍘熷MOT绫诲埆: {self.mot_original_classes}")
        print(f"[INFO] MOT绫诲埆鍚? {self.mot_class_names}")

        # 浣跨敤涓庤缁冭剼鏈浉鍚岀殑鍙傛暟閰嶇疆
        self.imgsz = 1280  # 涓庤缁冧繚鎸佷竴鑷?

        # 褰撳墠缁熶竴浣跨敤 BoxMOT-style 鍙ｅ緞锛歞etector conf = 0.10
        self.conf_threshold = 0.10
        self.iou_threshold = 0.7   # NMS鐨処oU闃堝€?
        self.max_det = 200         # 闄愬埗妫€娴嬫暟閲?

        print(f"[INFO] 鎺ㄧ悊鍙傛暟 - imgsz: {self.imgsz}, conf: {self.conf_threshold}, iou: {self.iou_threshold}, max_det: {self.max_det}")

        # 璁剧疆璺熻釜鍣ㄥ弬鏁帮紙鍩轰簬璁粌浼樺寲锛?
        self.tracker = tracker
        self.tracker_config = self._get_optimized_tracker_config()

        # 鏁版嵁闆嗚矾寰?
        self.root_dir = Path(root_dir)
        self.sequences_dir = self.root_dir / 'sequences'
        self.annotations_dir = self.root_dir / 'annotations'
        self.results = defaultdict(lambda: {'metrics': {}})

    def _auto_select_device(self, device):
        """鑷姩閫夋嫨璁＄畻璁惧 - 寮哄埗浣跨敤GPU"""
        if device == 'auto':
            device = 'cuda'  # 寮哄埗浣跨敤GPU

        if device == 'cuda':
            if torch.cuda.is_available():
                print(f"浣跨敤GPU: {torch.cuda.get_device_name(0)}")
                return 'cuda'
            else:
                print("CUDA涓嶅彲鐢紝鍒囨崲鍒癈PU妯″紡")
                return 'cpu'
        elif device == 'cpu':
            print("浣跨敤CPU妯″紡")
            return 'cpu'
        else:
            print(f"鏈煡璁惧: {device}, 浣跨敤CPU")
            return 'cpu'

    def _get_optimized_tracker_config(self):
        """Get the current unified ByteTrack configuration."""
        config = {
            'bytetrack.yaml': {
                'track_high_thresh': 0.45,
                'track_low_thresh': 0.10,
                'new_track_thresh': 0.45,
                'track_buffer': 25,
                'match_thresh': 0.8,
                'frame_rate': 30,
            }
        }
        return config.get(self.tracker, config['bytetrack.yaml'])

    def track_sequence(self, seq_name):
        """閫愬抚璺熻釜鍗曚釜搴忓垪锛堜紭鍖栫増锛?""
        # 鏋勫缓璺緞
        seq_dir = self.sequences_dir / seq_name
        gt_file = self.annotations_dir / f"{seq_name}.txt"

        if not seq_dir.exists():
            print(f"搴忓垪鐩綍涓嶅瓨鍦? {seq_dir}")
            return None

        if not gt_file.exists():
            print(f"GT鏂囦欢涓嶅瓨鍦? {gt_file}")
            return None

        # 鍔犺浇GT鏁版嵁
        gt_data = self._load_gt_file(str(gt_file))

        # 鑾峰彇鎵€鏈夊浘鍍忔枃浠?
        image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))

        if not image_files:
            print(f"搴忓垪涓病鏈夊浘鍍? {seq_dir}")
            return None

        # 浣跨敤鐢熸垚鍣ㄥ垎鎵瑰鐞?
        tracking_results = defaultdict(list)

        print(f"\n澶勭悊搴忓垪: {seq_name} ({len(image_files)} 甯?")
        print(f"浼樺寲妯″紡: {self.device}, imgsz: {self.imgsz}")

        # 鎵瑰鐞嗭紙鏈湴杩愯浣跨敤杈冨皬鐨刡atch size浠ヨ妭鐪佸唴瀛橈級
        batch_size = 1  # 鏈湴杩愯锛歜atch_size=1锛屼簯GPU杩愯锛歜atch_size=2
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc=f"鎵瑰鐞?{seq_name}"):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]

            # 鎵瑰鐞?
            for frame_idx, img_file in enumerate(batch_files, batch_start + 1):
                try:
                    # 浣跨敤涓庤缁冪浉鍚岀殑楂樺垎杈ㄧ巼鎺ㄧ悊
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

                    # 鎻愬彇璺熻釜缁撴灉
                    result = results[0]
                    if result.boxes is not None and hasattr(result.boxes, 'id'):
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # 锟斤拷鍙杢rack_id - 濡傛灉娌℃湁track_id鍒欑敓鎴愪竴涓?
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            else:
                                # 濡傛灉娌℃湁track_id锛屼娇鐢ㄦ娴嬫鐨勭储寮曚綔涓轰复鏃禝D
                                track_id = hash(f"{frame_idx}_{int(box.cls[0])}_{int(box.conf[0]*1000)}") % 1000000

                            conf = float(box.conf[0])
                            cls = int(box.cls[0])

                            # ===== 鍏抽敭淇锛氫娇鐢ㄧ被鍒悕鑰屼笉鏄疘D鏉ヨ繃婊?=====
                            # 閬垮厤鍘熷ID鍜孻OLO ID鐨勬槧灏勬贩娣?
                            if cls < len(self.model.names):
                                cls_name = self.model.names[cls]
                            else:
                                cls_name = f"unknown_{cls}"
                            
                            if cls_name not in self.mot_class_names:
                                # 涓嶅湪MOT绫诲埆涓紝杩囨护鎺?
                                continue

                            # 杞崲涓?MOT 鏍煎紡
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
                    print(f"澶勭悊绗?{frame_idx} 甯ф椂鍑洪敊: {e}")
                    continue

        # 浣跨敤 motmetrics 璁＄畻鎸囨爣
        metrics = self._compute_mot_metrics_with_motmetrics(tracking_results, gt_data, seq_name)

        self.results[seq_name] = {
            'metrics': metrics,
            'num_frames': len(image_files),
            'device': self.device
        }

        return metrics

    def _load_gt_file(self, file_path):
        """鍔犺浇VisDrone MOT鏍煎紡鐨凣T鏍囨敞鏂囦欢
        
        娉ㄦ剰锛欸T鏂囦欢涓殑绫诲埆ID鏄師濮媀isDrone鏍煎紡锛?-11锛?
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

                    # ===== 鍏抽敭淇锛氫娇鐢ㄥ師濮嬬被鍒獻D杩囨护GT =====
                    # GT涓殑cls_id鏄師濮媀isDrone鏍煎紡锛?-11锛?
                    # 鍙繚鐣橫OT鐩稿叧鐨勫師濮嬬被鍒?{1, 2, 4, 5, 6, 9}
                    if cls_id not in self.mot_original_classes:
                        # 涓嶆墦鍗拌鍛婏紝杩欎釜杩囨护鏄甯哥殑
                        continue

                    # ===== 鍏抽敭淇锛氭纭鐞唙isibility =====
                    # VisDrone鏍煎紡鐨刼cclusion鑼冨洿鏄?-2锛?=鏃犻伄鎸? 1=閮ㄥ垎閬尅, 2=涓ラ噸閬尅锛?
                    # 杞崲涓篗OT鏍煎紡鐨剉isibility锛?-1锛?
                    # 鏃犻伄鎸?0) -> 1.0, 閮ㄥ垎閬尅(1) -> 0.5, 涓ラ噸閬尅(2) -> 0.0
                    if occlusion == 0:
                        visibility = 1.0
                    elif occlusion == 1:
                        visibility = 0.5
                    else:  # occlusion == 2
                        visibility = 0.0

                    # ===== 杩囨护鏉′欢锛氬彧淇濈暀鏈夋晥鐨勬 =====
                    # 娉ㄦ剰锛歰cclusion=2锛堜弗閲嶉伄鎸★級鐨勬涓嶅簲淇濈暀锛坴isibility=0锛?
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
            print(f"鉂?璇诲彇GT鏂囦欢澶辫触: {e}")

        return data

    def _compute_mot_metrics_with_motmetrics(self, dt_results, gt_data, seq_name):
        """浣跨敤 motmetrics 搴撹绠楁爣鍑?MOT 鎸囨爣"""
        if not HAS_MOTMETRICS:
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        print(f"浣跨敤 motmetrics 璁＄畻鎸囨爣...")

        # 璋冭瘯淇℃伅
        print(f"DT缁撴灉甯ф暟: {len(dt_results)}")
        print(f"GT缁撴灉甯ф暟: {len(gt_data)}")

        # 缁熻鎬绘暟
        total_gt_boxes = sum(len(boxes) for boxes in gt_data.values())
        total_dt_boxes = sum(len(boxes) for boxes in dt_results.values())
        print(f"鎬籊T妗嗘暟: {total_gt_boxes}")
        print(f"鎬籇T妗嗘暟: {total_dt_boxes}")

        # 濡傛灉DT妗嗘暟杩囧锛屾樉锟斤拷锟借鍛?
        if total_dt_boxes > total_gt_boxes * 10:
            print(f"璀﹀憡: DT妗嗘暟({total_dt_boxes})杩滃ぇ浜嶨T妗嗘暟({total_gt_boxes})")
            print("杩欏彲鑳借〃绀虹被鍒繃婊ゅけ鏁堟垨缃俊搴﹂槇鍊艰繃浣?)

        # 鏁版嵁楠岃瘉
        if not gt_data:
            print("璀﹀憡: 娌℃湁GT鏁版嵁")
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        if not dt_results:
            print("璀﹀憡: 娌℃湁DT鏁版嵁")
            return self._compute_mot_metrics_manual(dt_results, gt_data)

        # 灏嗙粨鏋滆浆鎹负 motmetrics 鏍煎紡
        # GT 鏍煎紡: [frame_id, track_id, x, y, w, h, confidence, class_id, visibility]
        # DT 鏍煎紡: [frame_id, track_id, x, y, w, h, confidence]

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
                    box['visibility']  # 鉁?淇锛氫娇鐢ㄥ疄闄呰绠楃殑visibility
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

        print(f"杞崲鍚嶨T鏁伴噺: {len(gt_list)}")
        print(f"杞崲鍚嶥T鏁伴噺: {len(dt_list)}")

        try:
            # 鍒涘缓 accumulator - 浣跨敤姝ｇ‘鐨勯厤缃?
            acc = mm.MOTAccumulator(auto_id=False)

            # 鎸?frame_id 缁勭粐鏁版嵁
            frames = sorted(set([r[0] for r in gt_list] + [r[0] for r in dt_list]))
            print(f"澶勭悊甯ф暟: {len(frames)}")

            mot_events = []

            for frame_id in frames:
                # 鑾峰彇褰撳墠甯х殑GT
                gt_frame = [r for r in gt_list if r[0] == frame_id]
                # 鑾峰彇褰撳墠甯х殑DT
                dt_frame = [r for r in dt_list if r[0] == frame_id]

                if gt_frame and dt_frame:
                    # 鎻愬彇ID鍜屽潗鏍?
                    gt_ids = [r[1] for r in gt_frame]
                    dt_ids = [r[1] for r in dt_frame]

                    # 璁＄畻璺濈鐭╅樀 (浣跨敤1-IoU浣滀负璺濈锛岃繖鏄爣鍑嗙殑鍋氭硶)
                    cost_matrix = np.full((len(gt_frame), len(dt_frame)), np.inf)
                    for i, gt_r in enumerate(gt_frame):
                        for j, dt_r in enumerate(dt_frame):
                            # 璁＄畻IoU
                            gt_bbox = [gt_r[2], gt_r[3], gt_r[2] + gt_r[4], gt_r[3] + gt_r[5]]
                            dt_bbox = [dt_r[2], dt_r[3], dt_r[2] + dt_r[4], dt_r[3] + dt_r[5]]
                            iou = self._bbox_iou(gt_bbox, dt_bbox)

                            # 浣跨敤鏍囧噯鐨処oU璺濈璁＄畻锛氳窛绂?= 1 - IoU
                            # 杩欐牱IoU瓒婇珮锛岃窛绂昏秺灏忥紝绗﹀悎motmetrics鐨勮姹?
                            cost_matrix[i, j] = 1.0 - iou

                    # 鏇存柊accumulator
                    acc.update(gt_ids, dt_ids, cost_matrix, frameid=frame_id)

                elif gt_frame:
                    # 鍙湁GT - 浣滀负婕忔
                    gt_ids = [r[1] for r in gt_frame]
                    acc.update(gt_ids, [], [], frameid=frame_id)

                elif dt_frame:
                    # 鍙湁DT - 浣滀负璇
                    dt_ids = [r[1] for r in dt_frame]
                    acc.update([], dt_ids, [], frameid=frame_id)

            # 璁＄畻 MOT 鎸囨爣
            mh = mm.metrics.create()

            # 浣跨敤 motchallenge 鎸囨爣闆嗗悎
            metrics_to_compute = [
                'num_frames', 'num_objects', 'num_matches', 'num_false_positives',
                'num_misses', 'num_switches', 'mota', 'motp', 'idf1', 'precision',
                'recall', 'mostly_tracked', 'partially_tracked', 'mostly_lost'
            ]

            summary = mh.compute(acc, metrics=metrics_to_compute, name=seq_name)

            # 鎻愬彇缁撴灉 - 浣跨敤鏇村畨鍏ㄧ殑鏂瑰紡
            if summary is not None and not summary.empty:
                # 鑾峰彇绗竴琛岀殑缁撴灉
                row = summary.iloc[0]

                # 瀹夊叏鍦拌幏鍙栨寚鏍囧€硷紝浣跨敤float杞崲閬垮厤绫诲瀷閿欒
                mota = float(row.get('mota', 0)) * 100
                motp = float(row.get('motp', 0))
                idf1 = float(row.get('idf1', 0)) * 100

                # 鑾峰彇璇︾粏缁熻锛屼娇鐢╥nt杞崲
                num_fps = int(row.get('num_false_positives', 0))
                num_misses = int(row.get('num_misses', 0))
                num_switches = int(row.get('num_switches', 0))
                num_frames = int(row.get('num_frames', 0))

            else:
                print("motmetrics 璁＄畻缁撴灉涓虹┖锛屼娇鐢ㄦ墜鍔ㄨ绠?)
                return self._compute_mot_metrics_manual(dt_results, gt_data)

            print(f"=== Motmetrics 缁撴灉 ===")
            print(f"MOTA: {mota:.2f}%")
            print(f"MOTP: {motp:.3f}")
            print(f"IDF1: {idf1:.2f}%")
            print(f"FP: {num_fps}, Misses: {num_misses}, Switches: {num_switches}")
            print(f"澶勭悊甯ф暟: {num_frames}")

            # 璁＄畻TP - 浣跨敤棰勭粺璁＄殑鎬绘暟
            tp = total_gt_boxes - num_misses  # TP = GT - Misses
            fn = num_misses
            fp = num_fps

            print(f"TP: {tp}, FP: {fp}, FN: {fn}")

            # 璁＄畻Precision鍜孯ecall
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

            # 妫€鏌OTA鍊肩殑鍚堢悊鎬?
            if mota < -100 or mota > 100:
                print("璀﹀憡: MOTA鍊煎紓甯革紝鍙兘瀛樺湪鏁版嵁鏍煎紡闂")
                print(f"GT鎬绘暟: {total_gt_boxes}, DT鎬绘暟: {total_dt_boxes}")
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
            print(f"motmetrics 璁＄畻澶辫触: {e}")
            import traceback
            traceback.print_exc()
            print("浣跨敤鎵嬪姩璁＄畻鐨勬寚鏍?)
            return self._compute_mot_metrics_manual(dt_results, gt_data)

    def _compute_cost_matrix(self, gt_boxes, dt_boxes):
        """璁＄畻鍏宠仈浠ｄ环鐭╅樀"""
        cost_matrix = np.zeros((len(gt_boxes), len(dt_boxes)))

        for i, gt_box in enumerate(gt_boxes):
            for j, dt_box in enumerate(dt_boxes):
                # 浣跨敤 IoU 浣滀负鐩镐技搴︼紙璺濈 = 1 - IoU锛?
                iou = self._bbox_iou(
                    [gt_box['x'], gt_box['y'], gt_box['x'] + gt_box['w'], gt_box['y'] + gt_box['h']],
                    [dt_box['x'], dt_box['y'], dt_box['x'] + dt_box['w'], dt_box['y'] + dt_box['h']]
                )
                cost_matrix[i, j] = 1 - iou

        return cost_matrix

    def _compute_mot_metrics_manual(self, dt_results, gt_data):
        """鎵嬪姩璁＄畻 MOT 鎸囨爣锛堝鐢ㄧ増鏈級"""
        all_frames = sorted(set(list(dt_results.keys()) + list(gt_data.keys())))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # 璋冭瘯锛氭鏌ョ涓€甯х殑鍖归厤鎯呭喌
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

            # 璁＄畻鍖归厤
            matches = self._match_boxes(gt_boxes, dt_boxes)

            if frame_id == debug_frame:
                gt_boxes_list = gt_boxes
                dt_boxes_list = dt_boxes
                matches_list = matches

            total_tp += len(matches)
            total_fp += len(dt_boxes) - len(matches)
            total_fn += len(gt_boxes) - len(matches)

        # 璋冭瘯淇℃伅
        if debug_frame is not None:
            print(f"\n=== 璋冭瘯淇℃伅 (绗?{debug_frame} 甯? ===")
            print(f"GT妗嗘暟: {len(gt_boxes_list)}, DT妗嗘暟: {len(dt_boxes_list)}, 鍖归厤鏁? {len(matches_list)}")
            print(f"鍓嶅嚑涓狦T妗? {gt_boxes_list[:2] if gt_boxes_list else '鏃?}")
            print(f"鍓嶅嚑涓狣T妗? {dt_boxes_list[:2] if dt_boxes_list else '鏃?}")

        # 绠€鍖栫殑MOTA璁＄畻 - 闄愬埗鍦?-100%涔嬮棿
        if total_gt > 0:
            mota = 1 - (total_fp + total_fn) / total_gt
            motp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

            # 闄愬埗MOTA鍦?-100%涔嬮棿
            mota = max(0, min(1, mota))
        else:
            mota = 0
            motp = 0

        print(f"\n=== 璇︾粏鎸囨爣璁＄畻 ===")
        print(f"鎬昏 - GT: {total_gt}, TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"MOTA璁＄畻: 1 - ({total_fp} + {total_fn}) / {total_gt}")
        print(f"淇鍚庣殑MOTA: {mota * 100:.2f}%")
        print(f"MOTP璁＄畻: {total_tp} / ({total_tp} + {total_fp}) = {motp:.3f}")

        return {
            'MOTA': mota * 100,
            'MOTP': motp,
            'IDF1': 0,  # 绠€鍖栫増鏈?
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'ID_Switches': id_switches,
            'GT_Count': total_gt,
        }

    def _match_boxes(self, gt_boxes, dt_boxes):
        """鍖归厤妫€娴嬫鍜?GT"""
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
        """璁＄畻 IoU"""
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
        """璇勪及鎵€鏈夊簭鍒?""
        # 鑾峰彇鎵€鏈夊簭鍒?
        sequences = sorted([d.name for d in self.sequences_dir.iterdir() if d.is_dir()])

        all_metrics = {}

        print(f"\n鎵惧埌 {len(sequences)} 涓簭鍒?)
        print("="*70)

        for seq in sequences:
            try:
                metrics = self.track_sequence(seq)
                if metrics:
                    all_metrics[seq] = metrics
                    self._print_seq_metrics(seq, metrics)
            except Exception as e:
                print(f"澶勭悊搴忓垪 {seq} 鏃跺嚭閿? {e}")

        # 姹囨€荤粨鏋?
        self._print_summary(all_metrics)

        return all_metrics

    def _print_seq_metrics(self, seq_name, metrics):
        """鎵撳嵃鍗曚釜搴忓垪鐨勬寚鏍?""
        print(f"\n{seq_name}:")
        print(f"   MOTA: {metrics.get('MOTA', 0):6.2f}%  |  MOTP: {metrics.get('MOTP', 0):6.3f}")
        if 'IDF1' in metrics:
            print(f"   IDF1: {metrics['IDF1']:6.2f}%")
        print(f"   TP: {metrics.get('TP', 0):5d}  FP: {metrics.get('FP', 0):5d}  FN: {metrics.get('FN', 0):5d}")

    def _print_summary(self, all_metrics):
        """鎵撳嵃姹囨€荤粨鏋?""
        if not all_metrics:
            print("娌℃湁鏈夋晥鐨勮瘎浼扮粨鏋?)
            return

        # 璁＄畻骞冲潎鍊?
        mota_scores = [m.get('MOTA', 0) for m in all_metrics.values()]
        motp_scores = [m.get('MOTP', 0) for m in all_metrics.values()]
        idf1_scores = [m.get('IDF1', 0) for m in all_metrics.values() if 'IDF1' in m]

        avg_mota = np.mean(mota_scores)
        avg_motp = np.mean(motp_scores)
        avg_idf1 = np.mean(idf1_scores) if idf1_scores else 0

        print("\n" + "="*70)
        print("VisDrone MOT 璇勪及姹囨€伙紙璁粌鍙傛暟浼樺寲鐗堬級")
        print("="*70)

        print(f"\n鏁翠綋骞冲潎鎸囨爣 ({len(all_metrics)} 涓簭鍒?:")
        print(f"  MOTA:      {avg_mota:6.2f}%")
        print(f"  MOTP:      {avg_motp:6.3f}")
        print(f"  IDF1:      {avg_idf1:6.2f}%")

        print(f"\n鎸夊簭鍒楄缁嗙粨鏋?")
        print(f"{'搴忓垪鍚嶇О':<30} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8}")
        print("-"*75)
        for seq_name in sorted(all_metrics.keys()):
            m = all_metrics[seq_name]
            print(f"{seq_name:<30} {m.get('MOTA', 0):>7.2f}% {m.get('MOTP', 0):>7.3f} {m.get('IDF1', 0):>7.2f}%")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VisDrone-MOT 璇勪及锛堣缁冨弬鏁颁紭鍖栫増锛?)
    parser.add_argument('--model', type=str, required=True, help='YOLO妯″瀷璺緞 (e.g., best.pt)')
    parser.add_argument('--root_dir', type=str, required=True, help='VisDrone2019-MOT鏁版嵁闆嗘牴鐩綍')
    parser.add_argument('--device', type=str, default='auto', help='璁＄畻璁惧 (auto/cpu/cuda)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='璺熻釜鍣ㄩ厤缃?)
    parser.add_argument('--seq', type=str, default=None, help='璇勪及鍗曚釜搴忓垪锛堝彲閫夛級')
    parser.add_argument('--output', type=str, default='visdrone_mot_results.json', help='杈撳嚭JSON鏂囦欢')

    args = parser.parse_args()

    if not HAS_ULTRALYTICS:
        print("闇€瑕佸畨瑁?ultralytics")
        print("   pip install ultralytics")
        return

    print("="*70)
    print("VisDrone MOT 璇勪及宸ュ叿锛堣缁冨弬鏁颁紭鍖栫増锛?)
    print("="*70)
    print(f"妯″瀷: {args.model}")
    print(f"鏁版嵁: {args.root_dir}")
    print(f"璺熻釜鍣? {args.tracker}")
    print(f"璁惧: {args.device}")
    print("="*70)

    # 妫€鏌ユ暟鎹洰褰?
    root_path = Path(args.root_dir)
    if not (root_path / 'sequences').exists() or not (root_path / 'annotations').exists():
        print("鏁版嵁鐩綍缁撴瀯涓嶆纭紝闇€瑕佸寘鍚?")
        print("   鈹溾攢鈹€ sequences/  (鍥惧儚搴忓垪)")
        print("   鈹斺攢鈹€ annotations/  (GT鏍囨敞)")
        return

    evaluator = OptimizedVisDroneMOTEvaluator(args.model, args.root_dir, args.device, args.tracker)

    if args.seq:
        # 璇勪及鍗曚釜搴忓垪
        print(f"\n璇勪及鍗曚釜搴忓垪: {args.seq}\n")
        metrics = evaluator.track_sequence(args.seq)
        if metrics:
            evaluator._print_seq_metrics(args.seq, metrics)
    else:
        # 璇勪及鎵€鏈夊簭鍒?
        print(f"\n璇勪及鎵€鏈夊簭鍒梊n")
        all_metrics = evaluator.evaluate_all_sequences()

        # 淇濆瓨缁撴灉
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"缁撴灉宸蹭繚瀛樺埌: {args.output}\n")


if __name__ == '__main__':
    main()
