from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.mot.mot_utils import (
    bbox_iou_matrix,
    discover_sequences,
    get_image_extensions,
    get_yolo_to_mot_class_map,
    list_sequence_images,
    resolve_dataset_paths,
    write_mot_results,
    xyxy_to_ltwh,
)


def track_visdrone_mot_deepsort(
    model_path: str | Path,
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
    deepsort_root: str | Path = "../Yolov5_DeepSort_Pytorch",
    reid_ckpt: str | Path | None = None,
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    min_confidence: float = 0.1,
    nms_max_overlap: float = 1.0,
    max_dist: float = 0.2,
    max_iou_distance: float = 0.7,
    max_age: int = 70,
    n_init: int = 3,
    nn_budget: int = 100,
    match_iou: float = 0.3,
    verbose: bool = True,
) -> Path:
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    deepsort_root = Path(deepsort_root).resolve()
    if not deepsort_root.exists():
        raise FileNotFoundError(f"DeepSORT repo does not exist: {deepsort_root}")
    if str(deepsort_root) not in sys.path:
        sys.path.insert(0, str(deepsort_root))

    from deep_sort_pytorch.deep_sort import DeepSort

    if reid_ckpt is None:
        reid_ckpt = deepsort_root / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
    reid_ckpt = Path(reid_ckpt).resolve()
    if not reid_ckpt.exists():
        raise FileNotFoundError(f"DeepSORT ReID checkpoint does not exist: {reid_ckpt}")

    cfg, sequences_dir, _ = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    class_map = get_yolo_to_mot_class_map(cfg)
    sequence_dirs = discover_sequences(sequences_dir, image_extensions)
    if not sequence_dirs:
        raise ValueError(f"No image sequences found under {sequences_dir}")

    if output_dir is None:
        model_stem = Path(model_path).stem
        base_dir = Path("runs") / "mot" / f"{model_stem}_deepsort"
        output_dir = increment_path(base_dir, exist_ok=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    use_cuda = str(device).lower() != "cpu"

    for seq_dir in sequence_dirs:
        image_paths = list_sequence_images(seq_dir, image_extensions)
        if not image_paths:
            continue

        if verbose:
            print(f"[track-deepsort] {seq_dir.name}: {len(image_paths)} frames")

        tracker = DeepSort(
            model_path=str(reid_ckpt),
            max_dist=max_dist,
            min_confidence=min_confidence,
            nms_max_overlap=nms_max_overlap,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            use_cuda=use_cuda,
        )

        track_meta: dict[int, tuple[int, float]] = {}
        all_rows = []
        for frame_path in tqdm(image_paths, desc=seq_dir.name, disable=not verbose):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            predict_kwargs = {
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "verbose": False,
            }
            if device:
                predict_kwargs["device"] = device
            result = model.predict(frame, **predict_kwargs)[0]

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                tracker.increment_ages()
                continue

            det_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            det_conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1)
            det_cls = boxes.cls.cpu().numpy().astype(np.int32).reshape(-1)

            det_xywh = _xyxy_to_xywh(det_xyxy)
            outputs = tracker.update(det_xywh, det_conf, frame)
            rows = _deepsort_outputs_to_mot(
                outputs=outputs,
                det_xyxy=det_xyxy,
                det_conf=det_conf,
                det_cls=det_cls,
                frame_id=_safe_frame_id(frame_path),
                class_map=class_map,
                track_meta=track_meta,
                match_iou=match_iou,
            )
            if rows.size:
                all_rows.append(rows)

        merged_rows = np.vstack(all_rows) if all_rows else np.empty((0, 9), dtype=np.float32)
        write_mot_results(output_dir / f"{seq_dir.name}.txt", merged_rows, append=False)

    return output_dir


def _deepsort_outputs_to_mot(
    outputs: np.ndarray | list[np.ndarray],
    det_xyxy: np.ndarray,
    det_conf: np.ndarray,
    det_cls: np.ndarray,
    frame_id: int,
    class_map: dict[int, int],
    track_meta: dict[int, tuple[int, float]],
    match_iou: float,
) -> np.ndarray:
    if outputs is None or len(outputs) == 0:
        return np.empty((0, 9), dtype=np.float32)

    outputs = np.asarray(outputs, dtype=np.float32).reshape(-1, 5)
    track_xyxy = outputs[:, :4]
    track_ids = outputs[:, 4].astype(np.int32)

    matched_det_idx: dict[int, int] = {}
    if det_xyxy.size and track_xyxy.size:
        track_ltwh = xyxy_to_ltwh(track_xyxy)
        det_ltwh = xyxy_to_ltwh(det_xyxy)
        ious = bbox_iou_matrix(track_ltwh, det_ltwh)
        if ious.size:
            best_det = np.argmax(ious, axis=1)
            best_iou = ious[np.arange(len(track_ids)), best_det]
            for track_idx, det_idx in enumerate(best_det):
                if float(best_iou[track_idx]) >= float(match_iou):
                    matched_det_idx[int(track_idx)] = int(det_idx)

    rows = []
    ltwh = xyxy_to_ltwh(track_xyxy)
    for idx, track_id in enumerate(track_ids):
        if idx in matched_det_idx:
            det_idx = matched_det_idx[idx]
            yolo_class_id = int(det_cls[det_idx])
            conf = float(det_conf[det_idx])
            track_meta[int(track_id)] = (yolo_class_id, conf)
        elif int(track_id) in track_meta:
            yolo_class_id, conf = track_meta[int(track_id)]
        else:
            continue

        mot_class_id = int(class_map.get(int(yolo_class_id), int(yolo_class_id) + 1))
        rows.append(
            [
                int(frame_id),
                int(track_id),
                int(round(float(ltwh[idx, 0]))),
                int(round(float(ltwh[idx, 1]))),
                int(round(float(ltwh[idx, 2]))),
                int(round(float(ltwh[idx, 3]))),
                1,
                mot_class_id,
                float(conf),
            ]
        )

    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    xywh = np.empty((len(xyxy), 4), dtype=np.float32)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def _safe_frame_id(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track VisDrone MOT sequences with YOLO detector + external DeepSORT.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/visdrone_mot.yaml", help="Dataset YAML path.")
    parser.add_argument("--source", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", default=None, help="Directory to save MOT result txt files.")
    parser.add_argument("--deepsort-root", default="../Yolov5_DeepSort_Pytorch", help="External DeepSORT repo root.")
    parser.add_argument("--reid-ckpt", default=None, help="DeepSORT ReID checkpoint path.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Detection NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="", help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument("--min-confidence", type=float, default=0.1, help="DeepSORT minimum confidence.")
    parser.add_argument("--nms-max-overlap", type=float, default=1.0, help="DeepSORT NMS max overlap.")
    parser.add_argument("--max-dist", type=float, default=0.2, help="DeepSORT appearance distance threshold.")
    parser.add_argument("--max-iou-distance", type=float, default=0.7, help="DeepSORT IoU distance threshold.")
    parser.add_argument("--max-age", type=int, default=70, help="DeepSORT max age.")
    parser.add_argument("--n-init", type=int, default=3, help="DeepSORT confirm hits.")
    parser.add_argument("--nn-budget", type=int, default=100, help="DeepSORT appearance gallery size.")
    parser.add_argument("--match-iou", type=float, default=0.3, help="IoU threshold for assigning class/conf to tracks.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after tracking.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = track_visdrone_mot_deepsort(
        model_path=args.model,
        data=args.data,
        source=args.source,
        output_dir=args.output_dir,
        deepsort_root=args.deepsort_root,
        reid_ckpt=args.reid_ckpt,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        min_confidence=args.min_confidence,
        nms_max_overlap=args.nms_max_overlap,
        max_dist=args.max_dist,
        max_iou_distance=args.max_iou_distance,
        max_age=args.max_age,
        n_init=args.n_init,
        nn_budget=args.nn_budget,
        match_iou=args.match_iou,
        verbose=True,
    )
    print(f"[track-deepsort] results saved to {output_dir}")

    if args.eval:
        from tools.mot.eval_mot import eval_visdrone_mot

        eval_visdrone_mot(data=args.data, pred_dir=output_dir, source=args.source, verbose=True)


if __name__ == "__main__":
    main()
