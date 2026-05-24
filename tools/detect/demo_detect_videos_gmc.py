from __future__ import annotations

import argparse
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


def draw_stabilized_detections(frame, result, show_conf: bool = True) -> None:
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


def detect_videos_with_tracker(
    model_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    tracker: str | Path = "ultralytics/cfg/trackers/bytetrack_boxmot_gmc_color.yaml",
    conf: float = 0.10,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    show_conf: bool = True,
    realtime_show: bool = True,
    video_names: list[str] | None = None,
) -> list[Path]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(source_dir.glob("*.mp4"))
    if video_names:
        wanted = set(video_names)
        video_paths = [video_path for video_path in video_paths if video_path.name in wanted]
    if not video_paths:
        raise FileNotFoundError(f"No matching mp4 files found under {source_dir}")

    saved_paths: list[Path] = []

    for video_path in video_paths:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            fps = 15.0

        output_path = output_dir / video_path.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {output_path}")

        print(f"[demo] stabilized detect {video_path.name} -> {output_path}")
        track_kwargs = {
            "source": str(video_path),
            "stream": True,
            "persist": True,
            "tracker": str(tracker),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
            "classes": VISDRONE_CLASS_IDS,
            "save": False,
            "show": False,
        }
        if device:
            track_kwargs["device"] = device

        try:
            for result in model.track(**track_kwargs):
                frame = result.orig_img.copy()
                draw_stabilized_detections(frame, result, show_conf=show_conf)
                writer.write(frame)
                if realtime_show:
                    cv2.imshow("Stabilized Detection Demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            writer.release()
            if realtime_show:
                cv2.destroyAllWindows()

        saved_paths.append(output_path)

    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack + GMC demo without ID labels.")
    parser.add_argument("--model", default="weights/four-tripletattention.pt", help="Path to YOLO weights.")
    parser.add_argument("--source-dir", default="runs/demo/visdrone_testdev_top5_15fps", help="Directory containing mp4 files.")
    parser.add_argument("--output-dir", default="runs/demo/visdrone_testdev_top5_15fps_detected_gmc", help="Directory to save output videos.")
    parser.add_argument("--tracker", default="ultralytics/cfg/trackers/bytetrack_boxmot_gmc_color.yaml", help="Tracker YAML path.")
    parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence text.")
    parser.add_argument("--no-show", action="store_true", help="Disable realtime display window.")
    parser.add_argument("--video", action="append", default=None, help="Specific video file name(s) to process.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outputs = detect_videos_with_tracker(
        model_path=args.model,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show_conf=not args.hide_conf,
        realtime_show=not args.no_show,
        video_names=args.video,
    )
    print("[demo] saved files:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
