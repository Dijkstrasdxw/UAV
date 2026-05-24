from __future__ import annotations

import cv2

from ultralytics import YOLO


def main() -> None:
    model = YOLO("weights/four-tripletattention.pt")
    results = model.predict(
        source="VisDrone/images/test/9999947_00000_d_0000012.jpg",
        conf=0.25,
        imgsz=640,
        save=False,
        show=False,
    )

    plotted = results[0].plot()
    window_name = "Predict Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
