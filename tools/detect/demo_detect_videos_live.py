from __future__ import annotations

import cv2
import time
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


def draw_detections(frame, result, show_conf: bool = True) -> None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    for idx, (box, class_id) in enumerate(zip(xyxy, class_ids)):
        if class_id not in VISDRONE_CLASS_NAMES:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = VISDRONE_CLASS_COLORS[class_id]
        label = VISDRONE_CLASS_NAMES[class_id]
        if show_conf and confs is not None:
            label = f"{label} {confs[idx]:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 8
        if text_y < 18:
            text_y = min(frame.shape[0] - 8, y1 + 20)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def live_detect_video(
    model_path: str,
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    show_conf: bool = True,
    sync_to_video_fps: bool = False,
    display_fps: float | None = None,
    display_size: tuple[int, int] | None = (1280, 720),
) -> None:
    model = YOLO(model_path)
    target_frame_interval = None

    if display_fps is not None and display_fps > 0:
        target_frame_interval = 1.0 / display_fps
    elif sync_to_video_fps:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps and video_fps > 0:
                target_frame_interval = 1.0 / video_fps
        cap.release()

    predict_kwargs = {
        "source": video_path,
        "stream": True,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
        "classes": VISDRONE_CLASS_IDS,
        "save": False,
        "show": False,
    }
    if device:
        predict_kwargs["device"] = device

    last_display_time = None

    try:
        for result in model.predict(**predict_kwargs):
            frame = result.orig_img.copy()
            draw_detections(frame, result, show_conf=show_conf)
            if display_size is not None:
                frame = cv2.resize(frame, display_size)

            if target_frame_interval is not None and last_display_time is not None:
                elapsed = time.perf_counter() - last_display_time
                remaining = target_frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            cv2.imshow("Live Detection Demo", frame)
            last_display_time = time.perf_counter()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
