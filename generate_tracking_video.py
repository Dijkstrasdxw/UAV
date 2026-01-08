"""
生成VisDrone MOT视频，包含追踪边框和ID
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

"""
生成VisDrone MOT视频，包含追踪边框和ID（处理所有序列）
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

# 配置
SEQUENCES_DIR = Path('VisDrone/VisDrone2019-MOT-val/sequences')
OUTPUT_DIR = Path('res')
MODEL_PATH = 'runs/detect/visdrone_yolov8s16/weights/best.pt'

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# 类别颜色映射（不同类别用不同颜色）
CLASS_COLORS = {
    0: (0, 255, 0),         # pedestrian - 绿色
    1: (255, 0, 0),         # people - 蓝色
    2: (0, 255, 255),       # bicycle - 黄色
    3: (255, 0, 255),       # car - 洋红
    4: (128, 0, 255),       # van - 紫色
    5: (255, 128, 0),       # truck - 橙色
    6: (0, 128, 255),       # tricycle - 浅蓝
    7: (128, 255, 0),       # awning-tricycle - 浅绿
    8: (0, 0, 255),         # bus - 红色
    9: (200, 200, 0),       # motor - 深青
}

# MOT相关的类别
MOT_CLASS_NAMES = {"pedestrian", "people", "car", "van", "truck", "bus"}

def generate_video(seq_name):
    """生成追踪视频"""
    seq_dir = SEQUENCES_DIR / seq_name
    
    print(f"\n{'='*70}")
    print(f"处理序列: {seq_name}")
    print(f"{'='*70}")
    print(f"序列目录: {seq_dir}")
    
    # 获取所有图像文件
    image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))
    if not image_files:
        print(f"❌ 序列中没有图像")
        return False
    
    print(f"找到 {len(image_files)} 帧")
    
    # 加载YOLO模型
    model = YOLO(MODEL_PATH)
    print(f"模型加载成功")
    
    # 重置追踪器状态（处理新序列）
    if hasattr(model, 'predictor') and model.predictor is not None:
        model.predictor.reset_tracking()
    
    # 获取第一帧来确定视频尺寸
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"❌ 无法读取第一帧")
        return False
    
    h, w = first_frame.shape[:2]
    fps = 30
    
    # 创建视频写入器
    output_path = OUTPUT_DIR / f'{seq_name}_tracked.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"视频输出: {output_path}")
    print(f"分辨率: {w}x{h}, FPS: {fps}")
    
    # 追踪结果字典
    tracks_history = defaultdict(lambda: {'positions': [], 'cls': None})
    
    # 处理每一帧
    for frame_idx, img_file in enumerate(tqdm(image_files, desc="处理帧")):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue
        
        # YOLO检测和追踪
        results = model.track(
            frame,
            persist=True,
            conf=0.45,
            iou=0.7,
            imgsz=1280,
            verbose=False,
            device='cuda'
        )
        
        result = results[0]
        if result.boxes is not None and hasattr(result.boxes, 'id'):
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取追踪ID和置信度
                try:
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0].item()) if hasattr(box.id[0], 'item') else int(box.id[0])
                    else:
                        continue
                except (AttributeError, TypeError, IndexError):
                    continue
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # 获取类别名
                if cls < len(model.names):
                    cls_name = model.names[cls]
                else:
                    cls_name = f"unknown_{cls}"
                
                # 只绘制MOT相关类别
                if cls_name not in MOT_CLASS_NAMES:
                    continue
                
                # 记录追踪历史
                tracks_history[track_id]['positions'].append((x1, y1, x2, y2, frame_idx))
                tracks_history[track_id]['cls'] = cls
                
                # 获取颜色
                color = CLASS_COLORS.get(cls, (128, 128, 128))
                
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制追踪ID
                label = f'ID: {track_id}'
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_bg_pos = (x1, y1 - text_size[1] - 8)
                
                # 背景框
                cv2.rectangle(frame, 
                            (text_bg_pos[0], text_bg_pos[1]), 
                            (x1 + text_size[0], y1),
                            color, -1)
                
                # 文本
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制追踪轨迹（简化版）
        for track_id, data in tracks_history.items():
            if len(data['positions']) > 1:
                # 绘制最近5个位置的轨迹线
                positions = data['positions'][-5:]
                cls = data['cls']
                color = CLASS_COLORS.get(cls, (128, 128, 128))
                
                for i in range(len(positions) - 1):
                    x1_prev, y1_prev, x2_prev, y2_prev, _ = positions[i]
                    x1_curr, y1_curr, x2_curr, y2_curr, _ = positions[i + 1]
                    
                    center_prev = ((x1_prev + x2_prev) // 2, (y1_prev + y2_prev) // 2)
                    center_curr = ((x1_curr + x2_curr) // 2, (y1_curr + y2_curr) // 2)
                    
                    # 绘制简单的轨迹线
                    cv2.line(frame, center_prev, center_curr, color, 1)
        
        # 写入视频
        out.write(frame)
    
    # 释放视频写入器
    out.release()
    
    print(f"✅ {seq_name} 完成！")
    print(f"总帧数: {len(image_files)}")
    print(f"追踪轨迹数: {len(tracks_history)}")
    return True


def main():
    """批量处理所有序列"""
    print(f"{'='*70}")
    print(f"VisDrone MOT 视频生成工具")
    print(f"{'='*70}")
    print(f"数据目录: {SEQUENCES_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 获取所有序列
    sequences = sorted([d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()])
    
    if not sequences:
        print(f"❌ 未找到任何序列！")
        return
    
    print(f"找到 {len(sequences)} 个序列: {', '.join(sequences)}")
    
    # 处理每个序列
    success_count = 0
    for seq_name in sequences:
        try:
            if generate_video(seq_name):
                success_count += 1
        except Exception as e:
            print(f"❌ 处理 {seq_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"处理完成！成功处理 {success_count}/{len(sequences)} 个序列")
    print(f"视频保存在: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
