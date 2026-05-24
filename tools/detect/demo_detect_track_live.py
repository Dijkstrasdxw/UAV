from __future__ import annotations

import cv2
from ultralytics import YOLO

VISDRONE_CLASS_IDS = list(range(10))
VISDRONE_CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
VISDRONE_CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (0, 200, 0),
    4: (255, 165, 0),
    5: (0, 128, 255),
    6: (255, 0, 0),
    7: (180, 105, 255),
    8: (255, 0, 255),
    9: (128, 255, 0),
}


def draw_stabilized_detections(frame, result, show_conf: bool = True) -> None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    track_ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

    for idx, (box, class_id) in enumerate(zip(xyxy, class_ids)):
        if class_id not in VISDRONE_CLASS_NAMES:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = VISDRONE_CLASS_COLORS[class_id]
        label = VISDRONE_CLASS_NAMES[class_id]
        if track_ids is not None:
            label = f"ID {track_ids[idx]} {label}"
        if show_conf and confs is not None:
            label = f"{label} {confs[idx]:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 8
        if text_y < 18:
            text_y = min(frame.shape[0] - 8, y1 + 20)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def live_detect_with_tracker(
    model_path: str,
    video_path: str,
    tracker: str = "ultralytics/cfg/trackers/bytetrack_boxmot_gmc.yaml",
    conf: float = 0.10,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    show_conf: bool = True,
    display_size: tuple[int, int] | None = (1280, 720),
) -> None:
    model = YOLO(model_path)
    track_kwargs = {
        "source": video_path,
        "stream": True,
        "persist": True,
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
        "classes": VISDRONE_CLASS_IDS,
        "save": False,
        "show": False,
    }
    if device:
        track_kwargs["device"] = device

    try:
        for result in model.track(**track_kwargs):
            frame = result.orig_img.copy()
            draw_stabilized_detections(frame, result, show_conf=show_conf)
            if display_size is not None:
                frame = cv2.resize(frame, display_size)
            cv2.imshow("Live Detection+Track Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
