from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from tools.mot.mot_utils import discover_sequences, get_image_extensions, list_sequence_images, resolve_dataset_paths


def frames_to_videos(
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    source: str | Path | None = None,
    output_dir: str | Path = "runs/demo/visdrone_videos",
    fps: float = 25.0,
    limit: int | None = 5,
    sequence_names: list[str] | None = None,
    codec: str = "mp4v",
    verbose: bool = True,
) -> list[Path]:
    cfg, sequences_dir, _ = resolve_dataset_paths(data=data, source=source)
    image_extensions = get_image_extensions(cfg)
    sequence_dirs = discover_sequences(sequences_dir, image_extensions)
    if sequence_names:
        wanted = set(sequence_names)
        sequence_dirs = [seq_dir for seq_dir in sequence_dirs if seq_dir.name in wanted]
    elif limit is not None:
        sequence_dirs = sequence_dirs[:limit]

    if not sequence_dirs:
        raise ValueError(f"No matching sequences found under {sequences_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    saved_paths: list[Path] = []

    for seq_dir in sequence_dirs:
        image_paths = list_sequence_images(seq_dir, image_extensions)
        if not image_paths:
            continue

        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            raise ValueError(f"Failed to read first frame: {image_paths[0]}")
        height, width = first_frame.shape[:2]

        video_path = output_dir / f"{seq_dir.name}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        if verbose:
            print(f"[video] {seq_dir.name}: {len(image_paths)} frames -> {video_path}")

        try:
            for image_path in tqdm(image_paths, desc=seq_dir.name, disable=not verbose):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError(f"Failed to read frame: {image_path}")
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)
        finally:
            writer.release()

        saved_paths.append(video_path)

    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert VisDrone image sequences to mp4 videos.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/visdrone_mot.yaml", help="Dataset YAML path.")
    parser.add_argument("--source", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", default="runs/demo/visdrone_videos", help="Directory to save mp4 videos.")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video frame rate.")
    parser.add_argument("--limit", type=int, default=5, help="Number of sorted sequences to export.")
    parser.add_argument("--sequence", action="append", default=None, help="Specific sequence name(s) to export.")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, e.g. mp4v or XVID.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    video_paths = frames_to_videos(
        data=args.data,
        source=args.source,
        output_dir=args.output_dir,
        fps=args.fps,
        limit=args.limit,
        sequence_names=args.sequence,
        codec=args.codec,
        verbose=True,
    )
    print("[video] saved files:")
    for path in video_paths:
        print(path)


if __name__ == "__main__":
    main()
