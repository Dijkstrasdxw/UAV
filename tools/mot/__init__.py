from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.mot.eval_mot import eval_visdrone_mot
from tools.mot.track_mot import track_visdrone_mot


def run_visdrone_mot(
    model_path: str | Path,
    data: str | Path = "ultralytics/cfg/datasets/visdrone_mot.yaml",
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
    tracker: str | Path = "bytetrack.yaml",
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    eval_iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    pred_dir = track_visdrone_mot(
        model_path=model_path,
        data=data,
        source=source,
        output_dir=output_dir,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=verbose,
    )
    return eval_visdrone_mot(
        data=data,
        pred_dir=pred_dir,
        source=source,
        iou_threshold=eval_iou_threshold,
        verbose=verbose,
    )


__all__ = ["track_visdrone_mot", "eval_visdrone_mot", "run_visdrone_mot"]
