"""
VisDrone YOLOv8s 训练脚本
针对 RTX 4090 (24GB) GPU 优化
相对路径配置，可移植到任何环境
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

    # 获取项目根目录
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

# 训练配置（针对RTX 4090 24GB显存优化）
    print("\n开始训练VisDrone数据集（4090高性能配置）...")
    results = model.train(
        # 基本配置
        data=DATA_YAML,                  # ✅ 相对路径
        epochs=100,                      # 完整训练100个epoch
        imgsz=640,                       # 标准分辨率640

        # ✅ RTX 4090 24GB显存高性能配置
        batch=32,                        # 大batch=32，4090轻松承载

        # GPU和加速
        device=0,                        # GPU设备号
        half=True,                       # ✅ 启用混合精度（FP16），4090完全支持，加速训练
        scale=0.5,                       # 缩放比例
        flipud=0.0,                      # 上下翻转
        fliplr=0.5,                      # 左右翻转
        mosaic=1.0,                      # Mosaic增强

        # 训练策略
        patience=30,                     # 早停耐心值（无改进30个epoch停止）
        warmup_epochs=5,                 # 预热轮数
        warmup_momentum=0.8,             # 预热动量
        warmup_bias_lr=0.1,              # 预热偏置学习率

        # 保存和输出
        save=True,                       # 保存检查点
        save_period=10,                  # 每10个epoch保存一次
        verbose=True,                    # 显示训练进度

        # 结果保存
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),  # ✅ 相对路径
        name="visdrone_yolov8s",         # 实验名称

        # 其他 - ✅ 充分利用4090性能
        workers=8,                       # 8个进程加载数据（4090可以处理）
        close_mosaic=15,                 # 最后15个epoch关闭mosaic增强
        cache=True,                      # ✅ 启用缓存加速训练（24GB足够）
)

    # 验证
    print("\n验证模型...")
    metrics = model.val()

    # 测试
    print("\n模型训练完成！")
    best_weight_path = os.path.join(PROJECT_ROOT, "runs", "detect", "visdrone_yolov8s", "weights", "best.pt")
    print(f"最佳权重已保存到: {best_weight_path}")

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
