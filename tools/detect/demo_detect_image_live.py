from __future__ import annotations

from pathlib import Path

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


def live_detect_image(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    show_conf: bool = True,
    window_name: str = "Image Detection Demo",
    window_width: int | None = 1280,
    window_height: int | None = 720,
) -> None:
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=VISDRONE_CLASS_IDS,
        save=False,
        show=False,
        verbose=False,
    )

    if not results:
        raise RuntimeError("No detection result returned.")

    result = results[0]
    frame = result.orig_img.copy()
    draw_detections(frame, result, show_conf=show_conf)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if window_width and window_height:
        cv2.resizeWindow(window_name, window_width, window_height)
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_detect_image(
    model_path: str,
    image_path: str,
    output_path: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    show_conf: bool = True,
) -> str:
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=VISDRONE_CLASS_IDS,
        save=False,
        show=False,
        verbose=False,
    )

    if not results:
        raise RuntimeError("No detection result returned.")

    result = results[0]
    frame = result.orig_img.copy()
    draw_detections(frame, result, show_conf=show_conf)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), frame)
    if not ok:
        raise RuntimeError(f"Failed to save output image to: {output_path}")

    return str(output)
