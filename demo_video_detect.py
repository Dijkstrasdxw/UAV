from ultralytics import YOLO
import torch.multiprocessing as mp


MODEL_PATH = "weights/baseline.pt"
SOURCE = "runs/demo/visdrone_testdev_top5_15fps/uav0000355_00001_v.mp4"

def main():
    model = YOLO(MODEL_PATH)
    model.info()
    model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        save=False,
        save_txt=False,
        show=True,
        project="runs/demo",
        name="detect_video",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
