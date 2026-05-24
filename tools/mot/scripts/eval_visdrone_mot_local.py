"""
VisDrone MOT评估脚本 - 本地设备优化版 (适合 4GB 显存)
在VisDrone2019-MOT数据集上计算完整的MOT指标

优化特性：
- 分批处理，避免内存爆炸
- 低内存占用模式
- 显存优化（batch_size=1）
- 自动调整参数

使用���式：
    python eval_visdrone_mot_local.py --model best.pt --root_dir VisDrone2019-MOT-val
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
import cv2
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


class LocalVisDroneMOTEvaluator:
    """本地设备优化版评估器"""

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
        print(f"🚀 使用设备: {self.device}")

        # 加载模型，降低显存占用
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # 优化参数（根据显存调整）
        self.conf_threshold = 0.25 if self.device == 'cuda' else 0.3
        self.iou_threshold = 0.45 if self.device == 'cuda' else 0.5
        self.max_det = 300 if self.device == 'cuda' else 100

        print(f"📊 优化参数 - conf: {self.conf_threshold}, iou: {self.iou_threshold}, max_det: {self.max_det}")

        # 设置跟踪器参数（适合低显存）
        self.tracker = tracker
        self.tracker_config = self._optimize_tracker_config()

        # 数据集路径
        self.root_dir = Path(root_dir)
        self.sequences_dir = self.root_dir / 'sequences'
        self.annotations_dir = self.root_dir / 'annotations'
        self.results = defaultdict(lambda: {'metrics': {}})

    def _auto_select_device(self, device):
        """自动选择计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                # 检查显存大小
                if torch.cuda.get_device_properties(0).total_memory > 4 * 1024**3:  # > 4GB
                    return 'cuda'
                else:
                    print("⚠️ 显存不足，切换到 CPU 模式")
                    return 'cpu'
            return 'cpu'
        return device

    def _optimize_tracker_config(self):
        """优化跟踪器配置（适合低显存）"""
        config = {
            'bytetrack.yaml': {
                'track_high_thresh': 0.5 if self.device == 'cuda' else 0.3,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.6 if self.device == 'cuda' else 0.4,
                'track_buffer': 30 if self.device == 'cuda' else 10,
                'match_thresh': 0.8,
                'frame_rate': 30
            }
        }
        return config.get(self.tracker, config['bytetrack.yaml'])

    def track_sequence(self, seq_name):
        """逐帧跟踪单个序列（内存优化版）"""
        # 构建路径
        seq_dir = self.sequences_dir / seq_name
        gt_file = self.annotations_dir / f"{seq_name}.txt"

        if not seq_dir.exists():
            print(f"❌ 序列目录不存在: {seq_dir}")
            return None

        if not gt_file.exists():
            print(f"❌ GT文件不存在: {gt_file}")
            return None

        # 加载GT数据
        gt_data = self._load_gt_file(str(gt_file))

        # 获取所有图像文件（只读路径，不加载图像）
        image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))

        if not image_files:
            print(f"❌ 序列中没有图像: {seq_dir}")
            return None

        # 使用生成器分批处理，避免内存爆炸
        tracking_results = defaultdict(list)

        print(f"\n🎬 处理序列: {seq_name} ({len(image_files)} 帧)")
        print(f"📊 显存优化模式: {self.device}, conf: {self.conf_threshold}")

        # 分批处理
        batch_size = 1 if self.device == 'cuda' else 5
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc=f"批处理 {seq_name}"):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]

            # 批处理
            for frame_idx, img_file in enumerate(batch_files, batch_start + 1):
                # 内存优化：及时释放
                try:
                    # 直接处理图像路径，避免提前读取
                    results = self.model.track(
                        str(img_file),  # 使用路径而非图像数组
                        persist=True,
                        tracker=self.tracker,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_det,
                        verbose=False
                    )

                    # 提取跟踪结果
                    result = results[0]
                    if result.boxes is not None and hasattr(result.boxes, 'id'):
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # 提取track_id
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            else:
                                continue

                            conf = float(box.conf[0])
                            cls = int(box.cls[0])

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
                    print(f"⚠️ 处理第 {frame_idx} 帧时出错: {e}")
                    continue

        # 使用 motmetrics 计算指标（更标准）
        metrics = self._compute_mot_metrics_with_motmetrics(tracking_results, gt_data, seq_name)

        self.results[seq_name] = {
            'metrics': metrics,
            'num_frames': len(image_files),
            'device': self.device
        }

        return metrics

    def _load_gt_file(self, file_path):
        """加载VisDrone MOT格式的GT标注文件"""
        data = defaultdict(list)

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:  # VisDrone MOT 格式至少8个字段
                        continue

                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(parts[7]) if len(parts) > 7 else 0
                    visibility = float(parts[8]) if len(parts) > 8 else 1.0

                    # 过滤无效标注（visibility=0表示被遮挡，应该过滤）
                    if w > 0 and h > 0 and visibility > 0 and conf > 0:
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

        print(f"📊 使用 motmetrics 计算指标...")

        # 创建 motmetrics DataFrame
        df = mm.io.motmetrics_dataframe()

        # 构建事件日志
        events = []

        all_frames = sorted(set(list(dt_results.keys()) + list(gt_data.keys())))

        for frame_id in all_frames:
            gt_boxes = gt_data.get(frame_id, [])
            dt_boxes = dt_results.get(frame_id, [])

            # GT 框
            for gt_box in gt_boxes:
                events.append((frame_id, gt_box['track_id'], gt_box, 'GT'))

            # DT 框
            for dt_box in dt_boxes:
                events.append((frame_id, dt_box['track_id'], dt_box, 'FP'))

        # 转换为 motmetrics 格式
        acc = mm.MOTAccumulator(auto_id=True)

        for frame_id in all_frames:
            gt_boxes = gt_data.get(frame_id, [])
            dt_boxes = dt_results.get(frame_id, [])

            if gt_boxes and dt_boxes:
                # 计算距离矩阵
                cost_matrix = self._compute_cost_matrix(gt_boxes, dt_boxes)

                # 执行关联
                acc.update(
                    [box['track_id'] for box in gt_boxes],
                    [box['track_id'] for box in dt_boxes],
                    cost_matrix,
                    frameid=frame_id
                )
            elif gt_boxes:
                # 只有 GT，没有预测 -> FN
                acc.update(
                    [box['track_id'] for box in gt_boxes],
                    [],
                    [],
                    frameid=frame_id
                )
            elif dt_boxes:
                # 只有预测，没有 GT -> FP
                acc.update(
                    [],
                    [box['track_id'] for box in dt_boxes],
                    [],
                    frameid=frame_id
                )

        # 计算 MOT 指标
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_frames', 'num_objects',
                                          'num_false_positives', 'num_misses', 'num_switches'],
                            name=seq_name)

        # 提取结果
        mota = summary['mota'].iloc[0] * 100 if 'mota' in summary.columns else 0
        motp = summary['motp'].iloc[0] if 'motp' in summary.columns else 0
        idf1 = summary['idf1'].iloc[0] * 100 if 'idf1' in summary.columns else 0

        return {
            'MOTA': mota,
            'MOTP': motp,
            'IDF1': idf1,
            'num_frames': summary['num_frames'].iloc[0],
            'num_objects': summary['num_objects'].iloc[0],
            'num_false_positives': summary['num_false_positives'].iloc[0],
            'num_misses': summary['num_misses'].iloc[0],
            'num_switches': summary['num_switches'].iloc[0]
        }

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
        """手动计算 MOT 指标（备用方案）"""
        # 这里使用之前的实现
        all_frames = sorted(set(list(dt_results.keys()) + list(gt_data.keys())))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        for frame_id in all_frames:
            gt_boxes = gt_data.get(frame_id, [])
            dt_boxes = dt_results.get(frame_id, [])

            total_gt += len(gt_boxes)

            # 计算匹配
            matches = self._match_boxes(gt_boxes, dt_boxes)

            total_tp += len(matches)
            total_fp += len(dt_boxes) - len(matches)
            total_fn += len(gt_boxes) - len(matches)

        # 简化计算（没有 IDF1）
        mota = 1 - (total_fp + total_fn) / total_gt if total_gt > 0 else 0
        motp = 0.5  # 简化值

        return {
            'MOTA': max(0, mota * 100),
            'MOTP': motp,
            'IDF1': 0,  # 简化值
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

        print(f"\n🔍 找到 {len(sequences)} 个序列")
        print("="*70)

        for seq in sequences:
            try:
                metrics = self.track_sequence(seq)
                if metrics:
                    all_metrics[seq] = metrics
                    self._print_seq_metrics(seq, metrics)
            except Exception as e:
                print(f"❌ 处理序列 {seq} 时出错: {e}")

        # 汇总结果
        self._print_summary(all_metrics)

        return all_metrics

    def _print_seq_metrics(self, seq_name, metrics):
        """打印单个序列的指标"""
        print(f"\n✅ {seq_name}:")
        print(f"   MOTA: {metrics.get('MOTA', 0):6.2f}%  |  MOTP: {metrics.get('MOTP', 0):6.3f}")
        if 'IDF1' in metrics:
            print(f"   IDF1: {metrics['IDF1']:6.2f}%")
        print(f"   TP: {metrics.get('TP', 0):5d}  FP: {metrics.get('FP', 0):5d}  FN: {metrics.get('FN', 0):5d}")

    def _print_summary(self, all_metrics):
        """打印汇总结果"""
        if not all_metrics:
            print("❌ 没有有效的评估结果")
            return

        # 计算平均值
        mota_scores = [m.get('MOTA', 0) for m in all_metrics.values()]
        motp_scores = [m.get('MOTP', 0) for m in all_metrics.values()]
        idf1_scores = [m.get('IDF1', 0) for m in all_metrics.values() if 'IDF1' in m]

        avg_mota = np.mean(mota_scores)
        avg_motp = np.mean(motp_scores)
        avg_idf1 = np.mean(idf1_scores) if idf1_scores else 0

        print("\n" + "="*70)
        print("📈 VisDrone MOT 评估汇总 (本地优化版)")
        print("="*70)

        print(f"\n🎯 整体平均指标 ({len(all_metrics)} 个序列):")
        print(f"  MOTA:      {avg_mota:6.2f}%")
        print(f"  MOTP:      {avg_motp:6.3f}")
        print(f"  IDF1:      {avg_idf1:6.2f}%")

        print(f"\n📊 按序列详细结果:")
        print(f"{'序列名称':<30} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8}")
        print("-"*75)
        for seq_name in sorted(all_metrics.keys()):
            m = all_metrics[seq_name]
            print(f"{seq_name:<30} {m.get('MOTA', 0):>7.2f}% {m.get('MOTP', 0):>7.3f} {m.get('IDF1', 0):>7.2f}%")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VisDrone-MOT 评估 (本地优化版)')
    parser.add_argument('--model', type=str, required=True, help='YOLO模型路径 (e.g., best.pt)')
    parser.add_argument('--root_dir', type=str, required=True, help='VisDrone2019-MOT数据集根目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='跟踪器配置')
    parser.add_argument('--seq', type=str, default=None, help='评估单个序列（可选）')
    parser.add_argument('--output', type=str, default='visdrone_mot_results.json', help='输出JSON文件')

    args = parser.parse_args()

    if not HAS_ULTRALYTICS:
        print("❌ 需要安装 ultralytics")
        print("   pip install ultralytics")
        return

    print("="*70)
    print("🚀 VisDrone MOT 评估工具 (本地优化版)")
    print("="*70)
    print(f"📦 模型: {args.model}")
    print(f"📂 数据: {args.root_dir}")
    print(f"🔍 跟踪器: {args.tracker}")
    print(f"🎯 设备: {args.device}")
    print("="*70)

    # 检查数据目录
    root_path = Path(args.root_dir)
    if not (root_path / 'sequences').exists() or not (root_path / 'annotations').exists():
        print("❌ 数据目录结构不正确，需要包含:")
        print("   ├── sequences/  (图像序列)")
        print("   └── annotations/  (GT标注)")
        return

    evaluator = LocalVisDroneMOTEvaluator(args.model, args.root_dir, args.device, args.tracker)

    if args.seq:
        # 评估单个序列
        print(f"\n📍 评估单个序列: {args.seq}\n")
        metrics = evaluator.track_sequence(args.seq)
        if metrics:
            evaluator._print_seq_metrics(args.seq, metrics)
    else:
        # 评估所有序列
        print(f"\n📍 评估所有序列\n")
        all_metrics = evaluator.evaluate_all_sequences()

        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"✅ 结果已保存到: {args.output}\n")


if __name__ == '__main__':
    main()