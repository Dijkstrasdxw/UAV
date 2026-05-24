from __future__ import annotations

import torch.multiprocessing as mp

from tools.detect.demo_detect_videos_live import live_detect_video


def main() -> None:
    live_detect_video(
        model_path="weights/four-tripletattention.pt",
        video_path="runs/demo/visdrone_testdev_top5_15fps/uav0000201_00000_v.mp4",
        conf=0.25,
        iou=0.7,
        imgsz=640,
        device="0",
        show_conf=True,
        sync_to_video_fps=False,
        display_fps=15,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
