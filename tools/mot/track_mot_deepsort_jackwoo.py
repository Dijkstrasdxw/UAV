from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

from tools.mot.mot_utils import (
    discover_sequences,
    get_image_extensions,
    get_yolo_to_mot_class_map,
    list_sequence_images,
    resolve_dataset_paths,
    write_mot_results,
)


def track_visdrone_mot_jackwoo_deepsort(
    model_path: str | Path,
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
    jackwoo_root: str | Path = "third_party/Yolov7-tracker-v2.1",
    conf: float = 0.2,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    kalman_format: str = "byte",
    reid_model: str = "osnet_x0_25",
    reid_model_path: str | Path | None = None,
    reid_crop_size: tuple[int, int] = (128, 64),
    init_thresh: float = 0.3,
    track_buffer: int = 30,
    min_area: float = 150.0,
    verbose: bool = True,
) -> Path:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".yolo_cfg").resolve()))

    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    jackwoo_root = Path(jackwoo_root).resolve()
    tracker_root = jackwoo_root / "tracker"
    if not tracker_root.exists():
        raise FileNotFoundError(f"JackWoo tracker root does not exist: {tracker_root}")
    if str(tracker_root) not in sys.path:
        sys.path.insert(0, str(tracker_root))

    from trackers.deepsort_tracker import DeepSortTracker

    if reid_model_path is None:
        reid_model_path = jackwoo_root / "weights" / "osnet_x0_25.pth"
    reid_model_path = Path(reid_model_path).resolve()
    if not reid_model_path.exists():
        raise FileNotFoundError(f"ReID checkpoint does not exist: {reid_model_path}")

    cfg, sequences_dir, _ = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    class_map = get_yolo_to_mot_class_map(cfg)
    sequence_dirs = discover_sequences(sequences_dir, image_extensions)
    if not sequence_dirs:
        raise ValueError(f"No image sequences found under {sequences_dir}")

    if output_dir is None:
        model_stem = Path(model_path).stem
        base_dir = Path("runs") / "mot" / f"{model_stem}_jackwoo_deepsort"
        output_dir = increment_path(base_dir, exist_ok=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        device=str(device),
        trt=False,
        reid_model=str(reid_model),
        reid_model_path=str(reid_model_path),
        reid_crop_size=[int(reid_crop_size[0]), int(reid_crop_size[1])],
        kalman_format=str(kalman_format),
        conf_thresh=float(conf),
        conf_thresh_low=0.1,
        init_thresh=float(init_thresh),
        nms_thresh=float(iou),
        fuse_detection_score=False,
        track_buffer=int(track_buffer),
        gamma=0.1,
        min_area=float(min_area),
    )

    model = YOLO(model_path)

    for seq_dir in sequence_dirs:
        image_paths = list_sequence_images(seq_dir, image_extensions)
        if not image_paths:
            continue

        if verbose:
            print(f"[track-jackwoo-deepsort] {seq_dir.name}: {len(image_paths)} frames")

        tracker = DeepSortTracker(args=args, frame_rate=30)
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
                continue

            det_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            det_conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
            det_cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
            det_tlwh = _xyxy_to_tlwh(det_xyxy)
            output = np.concatenate([det_tlwh, det_conf, det_cls], axis=1)

            current_tracks = tracker.update(output, frame, frame)
            rows = _tracklets_to_mot_rows(
                tracklets=current_tracks,
                frame_id=_safe_frame_id(frame_path),
                class_map=class_map,
                min_area=min_area,
            )
            if rows.size:
                all_rows.append(rows)

        merged_rows = np.vstack(all_rows) if all_rows else np.empty((0, 9), dtype=np.float32)
        write_mot_results(output_dir / f"{seq_dir.name}.txt", merged_rows, append=False)

    return output_dir


def _xyxy_to_tlwh(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    tlwh = np.empty((len(xyxy), 4), dtype=np.float32)
    tlwh[:, 0] = xyxy[:, 0]
    tlwh[:, 1] = xyxy[:, 1]
    tlwh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    tlwh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return tlwh


def _tracklets_to_mot_rows(tracklets, frame_id: int, class_map: dict[int, int], min_area: float) -> np.ndarray:
    rows = []
    for trk in tracklets:
        bbox = np.asarray(trk.tlwh, dtype=np.float32)
        area = float(bbox[2] * bbox[3])
        if area <= float(min_area):
            continue
        yolo_class_id = int(trk.category)
        mot_class_id = int(class_map.get(yolo_class_id, yolo_class_id + 1))
        rows.append(
            [
                int(frame_id),
                int(trk.track_id),
                int(round(float(bbox[0]))),
                int(round(float(bbox[1]))),
                int(round(float(bbox[2]))),
                int(round(float(bbox[3]))),
                1,
                mot_class_id,
                float(trk.score),
            ]
        )
    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _safe_frame_id(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track VisDrone MOT with JackWoo-style DeepSORT and evaluate with local protocol.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/visdrone_mot.yaml", help="Dataset YAML path.")
    parser.add_argument("--source", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", default=None, help="Directory to save MOT txt files.")
    parser.add_argument("--jackwoo-root", default="third_party/Yolov7-tracker-v2.1", help="JackWoo tracker repo root.")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection/high confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument("--kalman-format", default="byte", help="Kalman format used by JackWoo tracker.")
    parser.add_argument("--reid-model", default="osnet_x0_25", help="ReID model name in JackWoo engine.")
    parser.add_argument("--reid-model-path", default=None, help="ReID checkpoint path.")
    parser.add_argument("--reid-crop-size", type=int, nargs=2, default=[128, 64], help="ReID crop size [h w].")
    parser.add_argument("--init-thresh", type=float, default=0.3, help="New track initialization threshold.")
    parser.add_argument("--track-buffer", type=int, default=30, help="Track buffer / max lost time.")
    parser.add_argument("--min-area", type=float, default=150.0, help="Minimum bbox area kept in results.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after tracking.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = track_visdrone_mot_jackwoo_deepsort(
        model_path=args.model,
        data=args.data,
        source=args.source,
        output_dir=args.output_dir,
        jackwoo_root=args.jackwoo_root,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        kalman_format=args.kalman_format,
        reid_model=args.reid_model,
        reid_model_path=args.reid_model_path,
        reid_crop_size=(args.reid_crop_size[0], args.reid_crop_size[1]),
        init_thresh=args.init_thresh,
        track_buffer=args.track_buffer,
        min_area=args.min_area,
        verbose=True,
    )
    print(f"[track-jackwoo-deepsort] results saved to {output_dir}")

    if args.eval:
        from tools.mot.eval_mot import eval_visdrone_mot

        eval_visdrone_mot(data=args.data, pred_dir=output_dir, source=args.source, verbose=True)


if __name__ == "__main__":
    main()
