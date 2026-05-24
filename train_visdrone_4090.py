from ultralytics import YOLO

# Current detection mainline: P2 + MKIR + PConv + CBAM.
model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml")
model.info()
model.train(
    data="./VisDrone.yaml",
    epochs=150,
    patience=25, 
    imgsz=640,
    batch=1,
    pretrained="yolo11s.pt",
    name="visdrone_yolo11",
    device=0,
    workers=2,
    half=True,
    cache=False,
)
