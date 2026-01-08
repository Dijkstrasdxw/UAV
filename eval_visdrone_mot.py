"""
VisDrone MOT评估脚本
在VisDrone2019-MOT数据集上计算完整的MOT指标

数据格式：
    VisDrone2019-MOT-val/
    ├── sequences/
    │   ├── uav0000086_00000_v/  (帧图像目录)
    │   │   ├── 0000001.jpg
    │   │   ├── 0000002.jpg
    │   │   └── ...
    │   └── ...
    └── annotations/  (GT标注)
        ├── uav0000086_00000_v.txt (MOT格式)
        └── ...

使用方式：
    # 评估全部序列
    python eval_visdrone_mot.py --model best.pt --root_dir VisDrone2019-MOT-val
    
    # 评估单个序列
    python eval_visdrone_mot.py --model best.pt --root_dir VisDrone2019-MOT-val --seq uav0000086_00000_v
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

try:
    import motmetrics as mm
    HAS_MOTMETRICS = True
except ImportError:
    HAS_MOTMETRICS = False
    warnings.warn("py-motmetrics not installed. For comprehensive MOT metrics, install: pip install motmetrics")


class VisDroneMOTEvaluator:
    """VisDrone MOT数据集评估器 - 支持图像序列输入"""
    
    def __init__(self, model_path, root_dir, device=None, tracker='bytetrack.yaml'):
        """初始化评估器
        
        Args:
            model_path (str): YOLO模型权重路径
            root_dir (str): VisDrone2019-MOT数据集根目录
            device (str): 计算设备
            tracker (str): 跟踪器配置
        """
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)
        
        self.root_dir = Path(root_dir)
        self.sequences_dir = self.root_dir / 'sequences'
        self.annotations_dir = self.root_dir / 'annotations'
        self.tracker = tracker
        self.results = defaultdict(lambda: {'metrics': {}})
    
    def track_sequence(self, seq_name):
        """逐帧跟踪单个序列
        
        Args:
            seq_name (str): 序列名称
            
        Returns:
            dict: 该序列的MOT指标
        """
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
        
        # 获取所有图像文件
        image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))
        
        if not image_files:
            print(f"❌ 序列中没有图像: {seq_dir}")
            return None
        
        # 逐帧跟踪和检测
        tracking_results = defaultdict(list)
        
        print(f"\n🎬 处理序列: {seq_name} ({len(image_files)} 帧)")
        
        for frame_idx, img_file in enumerate(tqdm(image_files, desc=f"跟踪 {seq_name}"), 1):
            # 读取图像
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"⚠️ 无法读取图像: {img_file}")
                continue
            
            # 运行跟踪 (persist=True 维持ID一致性)
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    conf=0.3,
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
                        
                        # 转换为YOLO格式 (x, y, w, h)
                        w = x2 - x1
                        h = y2 - y1
                        
                        tracking_results[frame_idx].append({
                            'track_id': track_id,
                            'x': x1,
                            'y': y1,
                            'w': w,
                            'h': h,
                            'conf': conf,
                            'cls': cls
                        })
            except Exception as e:
                print(f"❌ 处理第 {frame_idx} 帧时出错: {e}")
                continue
        
        # 计算MOT指标
        metrics = self._compute_mot_metrics(tracking_results, gt_data)
        
        self.results[seq_name] = {
            'metrics': metrics,
            'num_frames': len(image_files)
        }
        
        return metrics
    
    def _load_gt_file(self, file_path):
        """加载VisDrone MOT格式的GT标注文件
        
        标注格式：
        <frame_id>, <track_id>, <x>, <y>, <w>, <h>, <conf>, <class_id>, <visibility>, <occlusion>
        """
        data = defaultdict(list)
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 7:
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(parts[7]) if len(parts) > 7 else 0
                    visibility = float(parts[8]) if len(parts) > 8 else 1.0
                    
                    # VisDrone的GT标注条件
                    if w > 0 and h > 0 and visibility > 0 and conf > 0:
                        data[frame_id].append({
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
    
    def _compute_mot_metrics(self, dt_results, gt_data):
        """计算标准MOT指标
        
        MOTA = 1 - (FN + FP + IDSW) / GT
        MOTP = Σ(1-IoU) / 匹配数
        """
        all_frames = sorted(set(list(dt_results.keys()) + list(gt_data.keys())))
        
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou_error = 0
        total_matched = 0
        id_switches = 0
        
        # 用于追踪ID变化
        last_gt_id = defaultdict(lambda: None)
        
        for frame_id in all_frames:
            gt_boxes = gt_data.get(frame_id, [])
            dt_boxes = dt_results.get(frame_id, [])
            
            total_gt += len(gt_boxes)
            
            # 没有检测 → 全是FN
            if not dt_boxes:
                total_fn += len(gt_boxes)
                continue
            
            # 没有GT → 全是FP
            if not gt_boxes:
                total_fp += len(dt_boxes)
                continue
            
            # 计算IoU矩阵
            iou_matrix = self._compute_iou_matrix(gt_boxes, dt_boxes)
            
            # 贪心匹配 (按IoU从高到低)
            matched_gt = set()
            matched_dt = set()
            
            matches = []
            for i in range(len(gt_boxes)):
                for j in range(len(dt_boxes)):
                    if iou_matrix[i, j] > 0:
                        matches.append((i, j, iou_matrix[i, j]))
            
            matches.sort(key=lambda x: -x[2])  # 按IoU降序排列
            
            for i, j, iou in matches:
                if i not in matched_gt and j not in matched_dt and iou >= 0.5:
                    matched_gt.add(i)
                    matched_dt.add(j)
                    total_tp += 1
                    total_iou_error += (1 - iou)
                    total_matched += 1
                    
                    # 追踪ID一致性
                    gt_id = gt_boxes[i]['track_id']
                    dt_id = dt_boxes[j]['track_id']
                    
                    if last_gt_id[gt_id] is not None and last_gt_id[gt_id] != dt_id:
                        id_switches += 1
                    
                    last_gt_id[gt_id] = dt_id
            
            # 未匹配的都是错误
            total_fp += len(dt_boxes) - len(matched_dt)
            total_fn += len(gt_boxes) - len(matched_gt)
        
        # 计算最终指标
        mota = 1 - (total_fp + total_fn + id_switches) / total_gt if total_gt > 0 else 0
        motp = total_iou_error / total_matched if total_matched > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        return {
            'MOTA': max(0, mota * 100),  # MOTA通常表示为百分比，且不低于0
            'MOTP': motp,
            'Precision': precision,
            'Recall': recall,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'ID_Switches': id_switches,
            'GT_Count': total_gt,
        }
    
    def _compute_iou_matrix(self, gt_boxes, dt_boxes):
        """计算IoU矩阵"""
        iou_matrix = np.zeros((len(gt_boxes), len(dt_boxes)))
        
        for i, gt_box in enumerate(gt_boxes):
            gt_bbox = [gt_box['x'], gt_box['y'], 
                      gt_box['x'] + gt_box['w'], gt_box['y'] + gt_box['h']]
            
            for j, dt_box in enumerate(dt_boxes):
                dt_bbox = [dt_box['x'], dt_box['y'],
                          dt_box['x'] + dt_box['w'], dt_box['y'] + dt_box['h']]
                
                iou = self._bbox_iou(gt_bbox, dt_bbox)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    @staticmethod
    def _bbox_iou(box1, box2):
        """计算IoU"""
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
    
    def _save_tracking_results(self, tracking_results, output_file):
        """保存跟踪结果为MOT格式"""
        with open(output_file, 'w') as f:
            for frame_id in sorted(tracking_results.keys()):
                for box in tracking_results[frame_id]:
                    f.write(f"{frame_id},{box['track_id']},{box['x']:.1f},{box['y']:.1f},"
                           f"{box['w']:.1f},{box['h']:.1f},{box['conf']:.2f},{box['cls']},1\n")
    
    def evaluate_all_sequences(self):
        """评估所有序列"""
        # 获取所有序列
        sequences = sorted([d.name for d in self.sequences_dir.iterdir() if d.is_dir()])
        
        all_metrics = {}
        
        print(f"\n🔍 找到 {len(sequences)} 个序列")
        print("="*60)
        
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
        print(f"   MOTA: {metrics['MOTA']:6.2f}%  |  MOTP: {metrics['MOTP']:6.2f} px")
        print(f"   Prec: {metrics['Precision']:.4f}  |  Rec: {metrics['Recall']:.4f}")
        print(f"   TP: {metrics['TP']:5d}  FP: {metrics['FP']:5d}  FN: {metrics['FN']:5d}  ID_Sw: {metrics['ID_Switches']:3d}")
    
    def _print_summary(self, all_metrics):
        """打印汇总结果"""
        if not all_metrics:
            print("❌ 没有有效的评估结果")
            return
        
        # 计算平均值
        mota_scores = [m['MOTA'] for m in all_metrics.values()]
        motp_scores = [m['MOTP'] for m in all_metrics.values()]
        precision_scores = [m['Precision'] for m in all_metrics.values()]
        recall_scores = [m['Recall'] for m in all_metrics.values()]
        total_id_switches = sum([m['ID_Switches'] for m in all_metrics.values()])
        
        avg_mota = np.mean(mota_scores)
        avg_motp = np.mean(motp_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        
        print("\n" + "="*60)
        print("📈 VisDrone MOT 评估汇总")
        print("="*60)
        
        print(f"\n🎯 整体平均指标 ({len(all_metrics)} 个序列):")
        print(f"  MOTA:      {avg_mota:6.2f}%")
        print(f"  MOTP:      {avg_motp:6.2f} px")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  ID Switches: {total_id_switches}")
        
        print(f"\n📊 按序列详细结果:")
        print(f"{'序列名称':<30} {'MOTA':>8} {'MOTP':>8} {'Prec':>8} {'Rec':>8} {'ID_Sw':>6}")
        print("-"*70)
        for seq_name in sorted(all_metrics.keys()):
            m = all_metrics[seq_name]
            print(f"{seq_name:<30} {m['MOTA']:>7.2f}% {m['MOTP']:>7.2f}px {m['Precision']:>8.4f} {m['Recall']:>8.4f} {m['ID_Switches']:>6d}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VisDrone-MOT 评估')
    parser.add_argument('--model', type=str, required=True, help='YOLO模型路径 (e.g., best.pt)')
    parser.add_argument('--root_dir', type=str, required=True, help='VisDrone2019-MOT数据集根目录')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cpu/cuda/cuda:0)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='跟踪器配置 (bytetrack.yaml/botsort.yaml)')
    parser.add_argument('--seq', type=str, default=None, help='评估单个序列（可选）')
    parser.add_argument('--output', type=str, default='visdrone_mot_results.json', help='输出JSON文件')
    
    args = parser.parse_args()
    
    if not HAS_ULTRALYTICS:
        print("❌ 需要安装 ultralytics")
        print("   pip install ultralytics")
        return
    
    print("="*60)
    print("🚀 VisDrone MOT 评估工具")
    print("="*60)
    print(f"📦 模型: {args.model}")
    print(f"📂 数据: {args.root_dir}")
    print(f"🔍 跟踪器: {args.tracker}")
    print("="*60)
    
    # 检查数据目录
    root_path = Path(args.root_dir)
    if not (root_path / 'sequences').exists() or not (root_path / 'annotations').exists():
        print("❌ 数据目录结构不正确，需要包含:")
        print("   ├── sequences/  (图像序列)")
        print("   └── annotations/  (GT标注)")
        return
    
    evaluator = VisDroneMOTEvaluator(args.model, args.root_dir, args.device, args.tracker)
    
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
