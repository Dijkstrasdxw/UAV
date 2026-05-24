# test_visdrone.py

from ultralytics import YOLO

def main():
    # 你的验证逻辑
    model = YOLO("weights/p2+tripletattention.pt")  # 建议用绝对路径
    model.info()
    metrics = model.val(
        data="F:/code/python-code/ultralytics-main/VisDrone.yaml",
        split="test",
        imgsz=640,
        batch=4,
        workers=0,  # 可适当降低，Windows 对多进程不友好
        conf=0.001,
    )
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")


if __name__ == '__main__':
    # Windows 必须加这句！
    import torch.multiprocessing as mp

    mp.freeze_support()  # 可选，但推荐（防止打包成 exe 出错）

    ma
    in()
