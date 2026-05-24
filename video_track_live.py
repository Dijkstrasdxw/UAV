from __future__ import annotations

import torch.multiprocessing as mp

from tools.detect.demo_detect_track_live import live_detect_with_tracker


def main() -> None:
    model_path = "weights/four-tripletattention.pt"
    video_path = "F:/code/python-code/ultralytics-main/runs/demo/visdrone_testdev_top5_15fps/uav0000077_00720_v.mp4"
    tracker = "ultralytics/cfg/trackers/bytetrack_boxmot.yaml"
    conf = 0.10
    iou = 0.7
    imgsz = 640
    device = "0"
    show_conf = True

    live_detect_with_tracker(
        model_path=model_path,
        video_path=video_path,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        show_conf=show_conf,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
