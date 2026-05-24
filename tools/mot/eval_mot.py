from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from tools.mot.mot_utils import (
    bbox_ioa_matrix,
    bbox_iou_matrix,
    discover_sequences,
    get_distractor_classes,
    get_eval_classes,
    get_ignore_region_classes,
    get_ignore_region_ioa_threshold,
    get_image_extensions,
    get_occlusion_threshold,
    get_truncation_threshold,
    is_zero_mark_enabled,
    load_mot_predictions,
    load_visdrone_gt,
    resolve_dataset_paths,
)


SUMMARY_METRICS = [
    "num_frames",
    "num_objects",
    "mota",
    "motp",
    "idf1",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
]


def eval_visdrone_mot(
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    pred_dir: str | Path | None = None,
    source: str | Path | None = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg, sequences_dir, annotations_dir = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    eval_classes = get_eval_classes(cfg)
    distractor_class_ids = set(get_distractor_classes(cfg))
    ignore_region_class_ids = get_ignore_region_classes(cfg)
    ignore_region_ioa_threshold = get_ignore_region_ioa_threshold(cfg)
    zero_mark_enabled = is_zero_mark_enabled(cfg)
    truncation_threshold = get_truncation_threshold(cfg)
    occlusion_threshold = get_occlusion_threshold(cfg)
    sequence_dirs = discover_sequences(sequences_dir, image_extensions)
    if not sequence_dirs:
        raise ValueError(f"No image sequences found under {sequences_dir}")

    if pred_dir is None:
        raise ValueError("pred_dir is required for evaluation.")
    pred_dir = Path(pred_dir)

    metrics_handler = mm.metrics.create()
    class_results: dict[str, Any] = {}
    sequence_all_accs: dict[str, list[mm.MOTAccumulator]] = {seq_dir.name: [] for seq_dir in sequence_dirs}
    all_accs = []
    all_names = []

    for class_id, class_name in eval_classes.items():
        sequence_accs = []
        sequence_names = []
        for seq_dir in sequence_dirs:
            gt_path = annotations_dir / f"{seq_dir.name}.txt"
            pred_path = pred_dir / f"{seq_dir.name}.txt"
            acc = build_sequence_accumulator(
                gt_path=gt_path,
                pred_path=pred_path,
                class_id=class_id,
                distractor_class_ids=distractor_class_ids,
                ignore_region_class_ids=ignore_region_class_ids,
                zero_mark_enabled=zero_mark_enabled,
                truncation_threshold=truncation_threshold,
                occlusion_threshold=occlusion_threshold,
                iou_threshold=iou_threshold,
                ignore_region_ioa_threshold=ignore_region_ioa_threshold,
            )
            sequence_accs.append(acc)
            sequence_names.append(seq_dir.name)
            sequence_all_accs[seq_dir.name].append(acc)
            all_accs.append(acc)
            all_names.append(f"{class_name}:{seq_dir.name}")

        summary = metrics_handler.compute_many(
            sequence_accs,
            names=sequence_names,
            metrics=SUMMARY_METRICS,
            generate_overall=True,
        )
        combined = _row_to_metrics(summary.loc["OVERALL"])
        class_results[class_name] = {
            "combined": combined,
            "overall": combined,
            "per_sequence": {name: _row_to_metrics(summary.loc[name]) for name in sequence_names},
        }

    overall_summary = metrics_handler.compute_many(
        all_accs,
        names=all_names,
        metrics=SUMMARY_METRICS,
        generate_overall=True,
    )
    global_task = _row_to_metrics(overall_summary.loc["OVERALL"])
    sequence_global_results = {}
    for seq_name, seq_accs in sequence_all_accs.items():
        seq_summary = metrics_handler.compute_many(
            seq_accs,
            names=[f"{seq_name}:{idx}" for idx in range(len(seq_accs))],
            metrics=SUMMARY_METRICS,
            generate_overall=True,
        )
        sequence_global_results[seq_name] = _row_to_metrics(seq_summary.loc["OVERALL"])
    results = {
        "protocol": {
            "matching": "classwise",
            "aggregation": "global_task_sum_over_classes_and_sequences",
            "iou_threshold": float(iou_threshold),
            "ignore_region_ioa_threshold": float(ignore_region_ioa_threshold),
            "zero_mark_enabled": bool(zero_mark_enabled),
            "truncation_threshold": truncation_threshold,
            "occlusion_threshold": occlusion_threshold,
            "eval_class_ids": [int(x) for x in eval_classes],
            "distractor_class_ids": sorted(int(x) for x in distractor_class_ids),
            "ignore_region_class_ids": sorted(int(x) for x in ignore_region_class_ids),
        },
        "global_task": global_task,
        "overall": global_task,
        "sequences": sequence_global_results,
        "classes": class_results,
    }

    if verbose:
        _print_summary(results)
    return results


def build_sequence_accumulator(
    gt_path: str | Path,
    pred_path: str | Path,
    class_id: int,
    distractor_class_ids: set[int],
    ignore_region_class_ids: set[int],
    zero_mark_enabled: bool,
    truncation_threshold: int | float | None,
    occlusion_threshold: int | float | None,
    iou_threshold: float,
    ignore_region_ioa_threshold: float,
) -> mm.MOTAccumulator:
    gt_frames = load_visdrone_gt(gt_path)
    pred_frames = load_mot_predictions(pred_path)
    frame_ids = sorted(set(gt_frames) | set(pred_frames))
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in frame_ids:
        gt_valid, preds = _preprocess_frame(
            gt_objects=gt_frames.get(frame_id, []),
            pred_objects=pred_frames.get(frame_id, []),
            class_id=class_id,
            distractor_class_ids=distractor_class_ids,
            ignore_region_class_ids=ignore_region_class_ids,
            zero_mark_enabled=zero_mark_enabled,
            truncation_threshold=truncation_threshold,
            occlusion_threshold=occlusion_threshold,
            iou_threshold=iou_threshold,
            ignore_region_ioa_threshold=ignore_region_ioa_threshold,
        )

        gt_ids = [obj["track_id"] for obj in gt_valid]
        pred_ids = [obj["track_id"] for obj in preds]
        gt_boxes = np.asarray([obj["bbox"] for obj in gt_valid], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([obj["bbox"] for obj in preds], dtype=np.float32).reshape(-1, 4)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=iou_threshold)
        acc.update(gt_ids, pred_ids, distances)

    return acc


def _preprocess_frame(
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    class_id: int,
    distractor_class_ids: set[int],
    ignore_region_class_ids: set[int],
    zero_mark_enabled: bool,
    truncation_threshold: int | float | None,
    occlusion_threshold: int | float | None,
    iou_threshold: float,
    ignore_region_ioa_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relevant_gt_class_ids = {class_id, *distractor_class_ids}
    gt_candidates = [
        obj
        for obj in gt_objects
        if obj["class_id"] in relevant_gt_class_ids and _has_valid_bbox(obj["bbox"])
    ]
    ignore_regions = [
        obj for obj in gt_objects if obj["class_id"] in ignore_region_class_ids and _has_valid_bbox(obj["bbox"])
    ]
    preds = [obj for obj in pred_objects if obj["class_id"] == class_id and _has_valid_bbox(obj["bbox"])]

    remove_pred_indices: set[int] = set()
    unmatched_pred_indices = np.arange(len(preds), dtype=np.int32)

    if gt_candidates and preds:
        gt_boxes = np.asarray([obj["bbox"] for obj in gt_candidates], dtype=np.float32)
        pred_boxes = np.asarray([obj["bbox"] for obj in preds], dtype=np.float32)
        similarity_scores = bbox_iou_matrix(gt_boxes, pred_boxes)
        matching_scores = similarity_scores.copy()
        matching_scores[matching_scores < iou_threshold] = 0.0
        match_rows, match_cols = linear_sum_assignment(-matching_scores)
        matched_mask = matching_scores[match_rows, match_cols] > 0.0
        match_rows = match_rows[matched_mask]
        match_cols = match_cols[matched_mask]

        for row_idx, col_idx in zip(match_rows, match_cols):
            gt_obj = gt_candidates[int(row_idx)]
            if gt_obj["class_id"] in distractor_class_ids or _is_occluded_or_truncated(
                gt_obj, truncation_threshold, occlusion_threshold
            ):
                remove_pred_indices.add(int(col_idx))
        unmatched_pred_indices = np.setdiff1d(unmatched_pred_indices, match_cols.astype(np.int32), assume_unique=False)

    if ignore_regions and unmatched_pred_indices.size > 0:
        unmatched_boxes = np.asarray([preds[int(idx)]["bbox"] for idx in unmatched_pred_indices], dtype=np.float32)
        ignore_boxes = np.asarray([obj["bbox"] for obj in ignore_regions], dtype=np.float32)
        ioas = bbox_ioa_matrix(unmatched_boxes, ignore_boxes)
        if ioas.size > 0:
            covered_mask = np.any(ioas > ignore_region_ioa_threshold, axis=1)
            for idx in unmatched_pred_indices[covered_mask]:
                remove_pred_indices.add(int(idx))

    filtered_preds = [pred for idx, pred in enumerate(preds) if idx not in remove_pred_indices]
    gt_valid = [
        obj
        for obj in gt_candidates
        if obj["class_id"] == class_id
        and (not zero_mark_enabled or obj["mark"] != 0)
        and not _is_occluded_or_truncated(obj, truncation_threshold, occlusion_threshold)
    ]
    return gt_valid, filtered_preds


def _has_valid_bbox(bbox: np.ndarray) -> bool:
    return float(bbox[2]) > 0.0 and float(bbox[3]) > 0.0


def _is_occluded_or_truncated(
    obj: dict[str, Any], truncation_threshold: int | float | None, occlusion_threshold: int | float | None
) -> bool:
    if truncation_threshold is not None and obj["truncation"] > truncation_threshold:
        return True
    if occlusion_threshold is not None and obj["occlusion"] > occlusion_threshold:
        return True
    return False


def _row_to_metrics(row: Any) -> dict[str, float]:
    num_objects = float(row["num_objects"])
    mota = float(row["mota"]) * 100.0 if num_objects > 0 and not np.isinf(row["mota"]) else float("nan")
    recall = float(row["recall"]) * 100.0 if num_objects > 0 and not np.isnan(row["recall"]) else float("nan")
    return {
        "num_frames": float(row["num_frames"]),
        "GT": num_objects,
        "MOTA": mota,
        "MOTP": float(row["motp"]) * 100.0 if not np.isnan(row["motp"]) else float("nan"),
        "IDF1": float(row["idf1"]) * 100.0,
        "Precision": float(row["precision"]) * 100.0,
        "Recall": recall,
        "IDSW": float(row["num_switches"]),
        "FP": float(row["num_false_positives"]),
        "FN": float(row["num_misses"]),
    }


def _print_summary(results: dict[str, Any]) -> None:
    overall = results["global_task"]
    print("")
    print("=" * 96)
    print("VisDrone MOT Evaluation Summary")
    print("=" * 96)
    print(
        f"GLOBAL_TASK  MOTA {overall['MOTA']:.2f}  IDF1 {overall['IDF1']:.2f}  MOTP {overall['MOTP']:.2f}  "
        f"Precision {overall['Precision']:.2f}  Recall {overall['Recall']:.2f}  GT {int(overall['GT'])}  "
        f"IDSW {int(overall['IDSW'])}"
    )
    print("-" * 96)
    for class_name, class_result in results["classes"].items():
        metrics = class_result["overall"]
        print(
            f"{class_name:<18} MOTA {metrics['MOTA']:.2f}  IDF1 {metrics['IDF1']:.2f}  MOTP {metrics['MOTP']:.2f}  "
            f"Precision {metrics['Precision']:.2f}  Recall {metrics['Recall']:.2f}  GT {int(metrics['GT'])}  "
            f"IDSW {int(metrics['IDSW'])}"
        )
    print("-" * 96)
    for seq_name, metrics in results["sequences"].items():
        print(
            f"{seq_name:<18} MOTA {metrics['MOTA']:.2f}  IDF1 {metrics['IDF1']:.2f}  MOTP {metrics['MOTP']:.2f}  "
            f"Precision {metrics['Precision']:.2f}  Recall {metrics['Recall']:.2f}  GT {int(metrics['GT'])}  "
            f"IDSW {int(metrics['IDSW'])}"
        )
    print("=" * 96)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VisDrone MOT txt results with motmetrics.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/visdrone_mot.yaml", help="Dataset YAML path.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing one MOT txt per sequence.")
    parser.add_argument("--source", default=None, help="Optional dataset root override.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    eval_visdrone_mot(
        data=args.data,
        pred_dir=args.pred_dir,
        source=args.source,
        iou_threshold=args.iou_threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
