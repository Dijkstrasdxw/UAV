"""
VisDrone YOLOv8s 训练脚本 - 4090版本
优化配置：RTX 4090 (24GB显存) + Linux云端
"""

import os
from ultralytics import YOLO
import torch

def main():
    """主训练函数"""
    # 检查GPU
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")

    # 获取项目根目录（Linux兼容）
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(PROJECT_ROOT, "VisDrone.yaml")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "yolov8s.pt")

    # 验证文件是否存在
    print("\n检查必要文件...")
    if not os.path.exists(DATA_YAML):
        print(f"❌ 找不到数据集配置文件: {DATA_YAML}")
        print("   请确保 VisDrone.yaml 在项目根目录")
        exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 找不到预训练权重: {WEIGHTS_PATH}")
        exit(1)

    print(f"✅ 数据集配置: {DATA_YAML}")
    print(f"✅ 预训练权重: {WEIGHTS_PATH}")

    # 加载预训练模型
    print("\n加载YOLOv8s预训练模型...")
    model = YOLO(WEIGHTS_PATH)

    # 训练配置（针对RTX 4090 24GB显存 - 充分利用硬件）
    print("\n开始训练VisDrone数据集（4090高性能配置）...")
    results = model.train(
        # 基本配置
        data=DATA_YAML,                  # 数据集配置
        epochs=100,                      # ✅ 完整100个epoch
        imgsz=1280,                      # ✅ 高分辨率（小目标推荐1280）
        
        # 4090 24GB显存充分利用
        batch=32,                        # ✅ 大batch size（4090能承载）
        
        # GPU和加速
        device=0,                        # GPU设备号（Linux通常是0）
        half=True,                       # ✅ 启用混合精度（4090完全支持FP16）
        
        # 学习率配置
        lr0=0.01,                        # 初始学习率（可以更高）
        lrf=0.01,                        # 最终学习率比例
        momentum=0.937,                  # SGD动量
        weight_decay=0.0005,             # 权重衰减
        
        # 数据增强（针对小目标）
        hsv_h=0.015,                     # 色调变化
        hsv_s=0.7,                       # 饱和度变化
        hsv_v=0.4,                       # 亮度变化
        degrees=10.0,                    # 旋转角度
        translate=0.1,                   # 平移比例
        scale=0.5,                       # 缩放比例
        flipud=0.0,                      # 上下翻转
        fliplr=0.5,                      # 左右翻转
        mosaic=1.0,                      # Mosaic增强
        
        # 训练策略
        patience=30,                     # 早停耐心值
        warmup_epochs=5,                 # 预热轮数
        warmup_momentum=0.8,             # 预热动量
        warmup_bias_lr=0.1,              # 预热偏置学习率
        
        # 保存和输出
        save=True,                       # 保存检查点
        save_period=10,                  # 每10个epoch保存一次
        verbose=True,                    # 显示训练进度
        
        # 结果保存
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),  # 结果保存目录
        name="visdrone_yolov8s_4090",    # 实验名称
        
        # 其他 - 充分利用4090性能
        workers=8,                       # ✅ 多线程数据加载（4090有充足CPU资源）
        close_mosaic=15,                 # 最后15个epoch关闭mosaic增强
        cache='ram',                     # ✅ 缓存到内存（4090有足够内存）
        
        # 额外优化
        rect=True,                       # 矩形训练（加速）
        quad=True,                       # 四边形批处理（加速）
        multi_scale=True,                # 多尺度训练（更好的泛化）
    )

    # 验证
    print("\n验证模型...")
    metrics = model.val()

    # 测试
    print("\n模型训练完成！")
    best_weight_path = os.path.join(PROJECT_ROOT, "runs", "detect", "visdrone_yolov8s_4090", "weights", "best.pt")
    print(f"✅ 最佳权重已保存到: {best_weight_path}")

    # 可选：用最佳权重进行预测
    print("\n使用最佳模型进行测试...")
    best_model = YOLO(best_weight_path)

    # 你可以进行预测：
    # results = best_model.predict(source="test_image.jpg", conf=0.5)
    # 或者进行跟踪：
    # results = best_model.track(source="test_video.mp4")

# Windows多进程保护（必须有！）
if __name__ == '__main__':
    main()
