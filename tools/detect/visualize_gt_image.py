from __future__ import annotations

from pathlib import Path

import cv2

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


def _load_yolo_labels(label_path: str, image_width: int, image_height: int) -> list[tuple[int, int, int, int, int]]:
    labels: list[tuple[int, int, int, int, int]] = []
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            xc = float(parts[1]) * image_width
            yc = float(parts[2]) * image_height
            w = float(parts[3]) * image_width
            h = float(parts[4]) * image_height

            x1 = int(round(xc - w / 2))
            y1 = int(round(yc - h / 2))
            x2 = int(round(xc + w / 2))
            y2 = int(round(yc + h / 2))
            labels.append((class_id, x1, y1, x2, y2))

    return labels


def visualize_gt_image(
    image_path: str,
    label_path: str,
    output_path: str,
    line_width: int = 2,
    font_scale: float = 0.6,
) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    height, width = image.shape[:2]
    labels = _load_yolo_labels(label_path, image_width=width, image_height=height)

    for class_id, x1, y1, x2, y2 in labels:
        color = VISDRONE_CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = VISDRONE_CLASS_NAMES.get(class_id, f"class_{class_id}")

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

        text_y = y1 - 8
        if text_y < 18:
            text_y = min(height - 8, y1 + 20)

        cv2.putText(
            image,
            class_name,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"Failed to save output image to: {output_path}")

    return str(output)
