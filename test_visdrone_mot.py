from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.mot import run_visdrone_mot
from tools.mot.mot_utils import discover_sequences, get_image_extensions, resolve_dataset_paths


def prepare_subset_dataset(
    data: str | Path,
    source: str | Path | None,
    sequence_names: list[str] | None,
    max_frames: int | None,
    subset_root: str | Path,
) -> Path:
    cfg, sequences_dir, annotations_dir = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    all_sequences = discover_sequences(sequences_dir, image_extensions)
    selected = [seq for seq in all_sequences if not sequence_names or seq.name in sequence_names]
    if not selected:
        raise ValueError('No matching sequences found for subset generation.')

    subset_root = Path(subset_root)
    if subset_root.exists():
        shutil.rmtree(subset_root)
    (subset_root / 'sequences').mkdir(parents=True, exist_ok=True)
    (subset_root / 'annotations').mkdir(parents=True, exist_ok=True)

    for seq_dir in selected:
        dst_seq_dir = subset_root / 'sequences' / seq_dir.name
        dst_seq_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions], key=lambda p: int(p.stem))
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
        keep_frame_ids = {int(p.stem) for p in image_paths}
        for img_path in image_paths:
            shutil.copy2(img_path, dst_seq_dir / img_path.name)

        src_ann = annotations_dir / f'{seq_dir.name}.txt'
        dst_ann = subset_root / 'annotations' / src_ann.name
        with src_ann.open('r', encoding='utf-8') as fsrc, dst_ann.open('w', encoding='utf-8') as fdst:
            for line in fsrc:
                raw = line.strip()
                if not raw:
                    continue
                frame_id = int(float(raw.split(',', 1)[0]))
                if frame_id in keep_frame_ids:
                    fdst.write(raw + '\n')

    return subset_root


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run VisDrone MOT tracking + evaluation.')
    parser.add_argument('--model', default='weights/MKIR.pt', help='Path to YOLO weights.')
    parser.add_argument('--data', default='ultralytics/cfg/datasets/visdrone_mot.yaml', help='Dataset YAML path.')
    parser.add_argument('--source', default=None, help='Optional dataset root override.')
    parser.add_argument('--output-dir', default=None, help='Output directory for MOT txt results.')
    parser.add_argument('--tracker', default='bytetrack.yaml', help='Tracker YAML path.')
    parser.add_argument('--conf', type=float, default=0.1, help='Detection confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.7, help='Detection NMS IoU threshold.')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size.')
    parser.add_argument('--device', default='', help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument('--eval-iou-threshold', type=float, default=0.5, help='MOT matching IoU threshold.')
    parser.add_argument('--sequence', action='append', default=None, help='Only run specific sequence name(s).')
    parser.add_argument('--max-frames', type=int, default=None, help='Only keep the first N frames per selected sequence.')
    parser.add_argument('--subset-root', default='runs/mot_smoke_dataset', help='Temporary dataset root when using sequence/max-frames filtering.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    source = args.source
    if args.sequence or args.max_frames is not None:
        source = prepare_subset_dataset(
            data=args.data,
            source=args.source,
            sequence_names=args.sequence,
            max_frames=args.max_frames,
            subset_root=args.subset_root,
        )
        print(f'[subset] prepared temporary dataset at {source}')

    results = run_visdrone_mot(
        model_path=args.model,
        data=args.data,
        source=source,
        output_dir=args.output_dir,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        eval_iou_threshold=args.eval_iou_threshold,
        verbose=True,
    )

    print('[eval] overall metrics:')
    print(json.dumps(results['overall'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
