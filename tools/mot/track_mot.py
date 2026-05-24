from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.mot.mot_utils import (
    convert_results_to_mot,
    discover_sequences,
    get_image_extensions,
    get_yolo_to_mot_class_map,
    list_sequence_images,
    resolve_dataset_paths,
    write_mot_results,
)


def track_visdrone_mot(
    model_path: str | Path,
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
    tracker: str | Path = "bytetrack.yaml",
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    verbose: bool = True,
) -> Path:
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    cfg, sequences_dir, _ = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    class_map = get_yolo_to_mot_class_map(cfg)
    sequence_dirs = discover_sequences(sequences_dir, image_extensions)
    if not sequence_dirs:
        raise ValueError(f"No image sequences found under {sequences_dir}")

    if output_dir is None:
        model_stem = Path(model_path).stem
        base_dir = Path("runs") / "mot" / f"{model_stem}_bytetrack"
        output_dir = increment_path(base_dir, exist_ok=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in sequence_dirs:
        image_paths = list_sequence_images(seq_dir, image_extensions)
        if not image_paths:
            continue

        if verbose:
            print(f"[track] {seq_dir.name}: {len(image_paths)} frames")

        model = YOLO(model_path)
        all_rows = []
        for frame_path in tqdm(image_paths, desc=seq_dir.name, disable=not verbose):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            track_kwargs = {
                "persist": True,
                "tracker": str(tracker),
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "verbose": False,
            }
            if device:
                track_kwargs["device"] = device

            results = model.track(frame, **track_kwargs)
            rows = convert_results_to_mot(results[0], frame_id=_safe_frame_id(frame_path), class_map=class_map)
            if rows.size:
                all_rows.append(rows)

        merged_rows = np.vstack(all_rows) if all_rows else np.empty((0, 9), dtype=np.float32)
        write_mot_results(output_dir / f"{seq_dir.name}.txt", merged_rows, append=False)

    return output_dir


def _safe_frame_id(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track VisDrone MOT sequences with Ultralytics YOLO + ByteTrack.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/visdrone_mot.yaml", help="Dataset YAML path.")
    parser.add_argument("--source", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", default=None, help="Directory to save MOT result txt files.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker YAML path.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="", help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after tracking.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = track_visdrone_mot(
        model_path=args.model,
        data=args.data,
        source=args.source,
        output_dir=args.output_dir,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=True,
    )
    print(f"[track] results saved to {output_dir}")

    if args.eval:
        from tools.mot.eval_mot import eval_visdrone_mot

        eval_visdrone_mot(data=args.data, pred_dir=output_dir, source=args.source, verbose=True)


if __name__ == "__main__":
    main()
