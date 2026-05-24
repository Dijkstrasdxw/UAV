from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


DEFAULT_DATASET_CFG = Path("ultralytics/cfg/datasets/visdrone_mot.yaml")


def load_dataset_cfg(data: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    if data is None:
        data = DEFAULT_DATASET_CFG
    if isinstance(data, dict):
        return data
    path = Path(data)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_paths(
    data: str | Path | dict[str, Any] | None = None, source: str | Path | None = None
) -> tuple[dict[str, Any], Path, Path]:
    cfg = load_dataset_cfg(data)
    benchmark = cfg.get("benchmark", {})
    root = Path(source) if source is not None else Path(cfg["path"])
    sequences_dir = root / benchmark.get("split", "sequences")
    annotations_dir = root / benchmark.get("annotations", "annotations")
    return cfg, sequences_dir, annotations_dir


def get_image_extensions(cfg: dict[str, Any]) -> tuple[str, ...]:
    benchmark = cfg.get("benchmark", {})
    image_extensions = benchmark.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp"])
    return tuple(str(x).lower() for x in image_extensions)


def get_eval_classes(cfg: dict[str, Any]) -> dict[int, str]:
    eval_classes = cfg.get("benchmark", {}).get("eval_classes", {})
    return {int(k): str(v) for k, v in eval_classes.items()}


def get_distractor_classes(cfg: dict[str, Any]) -> dict[int, str]:
    distractor_classes = cfg.get("benchmark", {}).get("distractor_classes", {})
    return {int(k): str(v) for k, v in distractor_classes.items()}


def get_ignore_region_classes(cfg: dict[str, Any]) -> set[int]:
    benchmark = cfg.get("benchmark", {})
    ignore_region_classes = benchmark.get("ignore_region_classes", [])
    return {int(x) for x in ignore_region_classes}


def is_zero_mark_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("benchmark", {}).get("zero_mark", True))


def get_truncation_threshold(cfg: dict[str, Any]) -> int | float | None:
    truncation = cfg.get("benchmark", {}).get("truncation", {})
    if not truncation or not truncation.get("enabled", False):
        return None
    return truncation.get("thresh", 0)


def get_occlusion_threshold(cfg: dict[str, Any]) -> int | float | None:
    occlusion = cfg.get("benchmark", {}).get("occlusion", {})
    if not occlusion or not occlusion.get("enabled", False):
        return None
    return occlusion.get("thresh", 0)


def get_ignore_region_ioa_threshold(cfg: dict[str, Any]) -> float:
    crowd_ignore = cfg.get("benchmark", {}).get("crowd_ignore_region", {})
    return float(crowd_ignore.get("ioa_threshold", 0.5))


def get_yolo_to_mot_class_map(cfg: dict[str, Any]) -> dict[int, int]:
    class_offset = int(cfg.get("benchmark", {}).get("class_offset", 1))
    names = cfg.get("names", {})
    return {int(k): int(k) + class_offset for k in sorted(int(x) for x in names)}


def discover_sequences(sequences_dir: str | Path, image_extensions: tuple[str, ...]) -> list[Path]:
    sequences_dir = Path(sequences_dir)
    if not sequences_dir.exists():
        raise FileNotFoundError(f"Sequences directory does not exist: {sequences_dir}")

    sequence_dirs = []
    for seq_dir in sorted(x for x in sequences_dir.iterdir() if x.is_dir()):
        if any(p.suffix.lower() in image_extensions for p in seq_dir.iterdir() if p.is_file()):
            sequence_dirs.append(seq_dir)
    return sequence_dirs


def list_sequence_images(seq_dir: str | Path, image_extensions: tuple[str, ...]) -> list[Path]:
    seq_dir = Path(seq_dir)
    images = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    images.sort(key=lambda p: _frame_id_from_path(p))
    return images


def _frame_id_from_path(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def xyxy_to_ltwh(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    ltwh = xyxy.copy()
    ltwh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    ltwh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return ltwh


def load_visdrone_gt(txt_path: str | Path) -> dict[int, list[dict[str, Any]]]:
    txt_path = Path(txt_path)
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not txt_path.exists():
        raise FileNotFoundError(f"GT file does not exist: {txt_path}")

    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 10:
                continue
            frame_id = int(float(parts[0]))
            target_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            mark = int(float(parts[6]))
            class_id = int(float(parts[7]))
            truncation = int(float(parts[8]))
            occlusion = int(float(parts[9]))
            frames[frame_id].append(
                {
                    "track_id": target_id,
                    "bbox": np.array([left, top, width, height], dtype=np.float32),
                    "class_id": class_id,
                    "mark": mark,
                    "truncation": truncation,
                    "occlusion": occlusion,
                }
            )
    return dict(frames)


def load_mot_predictions(txt_path: str | Path) -> dict[int, list[dict[str, Any]]]:
    txt_path = Path(txt_path)
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not txt_path.exists():
        return {}

    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            class_id = int(float(parts[7]))
            confidence = float(parts[8])
            frames[frame_id].append(
                {
                    "track_id": track_id,
                    "bbox": np.array([left, top, width, height], dtype=np.float32),
                    "class_id": class_id,
                    "confidence": confidence,
                }
            )
    return dict(frames)


def convert_results_to_mot(
    results: "Results" | np.ndarray | None, frame_id: int, class_map: dict[int, int] | None = None
) -> np.ndarray:
    if results is None:
        return np.empty((0, 9), dtype=np.float32)

    if class_map is None:
        class_map = {}

    if isinstance(results, np.ndarray):
        if results.size == 0:
            return np.empty((0, 9), dtype=np.float32)
        boxes = results[:, :4]
        track_ids = results[:, 4].astype(np.int32)
        confs = results[:, 5].astype(np.float32)
        classes = results[:, 6].astype(np.int32)
    else:
        boxes_obj = results.boxes
        if boxes_obj is None or boxes_obj.id is None or len(boxes_obj) == 0:
            return np.empty((0, 9), dtype=np.float32)
        boxes = boxes_obj.xyxy.cpu().numpy()
        track_ids = boxes_obj.id.cpu().numpy().astype(np.int32).reshape(-1)
        confs = boxes_obj.conf.cpu().numpy().astype(np.float32).reshape(-1)
        classes = boxes_obj.cls.cpu().numpy().astype(np.int32).reshape(-1)

    ltwh = xyxy_to_ltwh(boxes)
    rows = []
    for idx in range(len(track_ids)):
        yolo_class_id = int(classes[idx])
        mot_class_id = int(class_map.get(yolo_class_id, yolo_class_id + 1))
        rows.append(
            [
                int(frame_id),
                int(track_ids[idx]),
                int(round(float(ltwh[idx, 0]))),
                int(round(float(ltwh[idx, 1]))),
                int(round(float(ltwh[idx, 2]))),
                int(round(float(ltwh[idx, 3]))),
                1,
                mot_class_id,
                float(confs[idx]),
            ]
        )
    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def write_mot_results(txt_path: str | Path, mot_rows: np.ndarray, append: bool = False) -> None:
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if mot_rows is None or mot_rows.size == 0:
        if not append:
            txt_path.write_text("", encoding="utf-8")
        return

    mode = "a" if append else "w"
    with txt_path.open(mode, encoding="utf-8") as f:
        np.savetxt(f, mot_rows, fmt="%d,%d,%d,%d,%d,%d,%d,%d,%.6f")


def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a_x1 = boxes_a[:, 0:1]
    a_y1 = boxes_a[:, 1:2]
    a_x2 = a_x1 + boxes_a[:, 2:3]
    a_y2 = a_y1 + boxes_a[:, 3:4]

    b_x1 = boxes_b[:, 0][None, :]
    b_y1 = boxes_b[:, 1][None, :]
    b_x2 = b_x1 + boxes_b[:, 2][None, :]
    b_y2 = b_y1 + boxes_b[:, 3][None, :]

    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2:3]) * np.maximum(0.0, boxes_a[:, 3:4])
    area_b = np.maximum(0.0, boxes_b[:, 2][None, :]) * np.maximum(0.0, boxes_b[:, 3][None, :])
    union = area_a + area_b - inter_area
    return np.where(union > 0.0, inter_area / union, 0.0)


def bbox_ioa_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a_x1 = boxes_a[:, 0:1]
    a_y1 = boxes_a[:, 1:2]
    a_x2 = a_x1 + boxes_a[:, 2:3]
    a_y2 = a_y1 + boxes_a[:, 3:4]

    b_x1 = boxes_b[:, 0][None, :]
    b_y1 = boxes_b[:, 1][None, :]
    b_x2 = b_x1 + boxes_b[:, 2][None, :]
    b_y2 = b_y1 + boxes_b[:, 3][None, :]

    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2:3]) * np.maximum(0.0, boxes_a[:, 3:4])
    return np.where(area_a > 0.0, inter_area / area_a, 0.0)


def suppress_distractor_predictions(
    predictions: list[dict[str, Any]], distractors: list[dict[str, Any]], iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    if not predictions or not distractors:
        return predictions

    pred_boxes = np.stack([pred["bbox"] for pred in predictions], axis=0)
    distractor_boxes = np.stack([gt["bbox"] for gt in distractors], axis=0)
    ious = bbox_iou_matrix(distractor_boxes, pred_boxes)
    if ious.size == 0:
        return predictions

    cost = 1.0 - ious
    cost[ious < iou_threshold] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)
    remove_cols = {c for r, c in zip(row_ind, col_ind) if ious[r, c] >= iou_threshold}
    return [pred for idx, pred in enumerate(predictions) if idx not in remove_cols]
