"""Smoke-test the inference path: RGB image + LiDAR BEV + radar BEV -> boxes.

Uses whatever checkpoint is in logs/ and the synthetic sample 000000.
Run from the repository root:  python tools/test_predict.py
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.getcwd())
from detector import YOLO  # noqa: E402

SAMPLE = "000000"
ROOT = os.path.join("VOCdevkit", "VOC2007")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "logs/best_epoch_weights.pth"
    if not os.path.exists(model_path):
        raise SystemExit("checkpoint not found: %s (train one epoch first)" % model_path)

    yolo = YOLO(model_path=model_path, classes_path="model_data/classes.txt",
                confidence=0.05, cuda=True)

    image = Image.open(os.path.join(ROOT, "JPEGImages", SAMPLE + ".jpg"))
    lidar = Image.open(os.path.join(ROOT, "LRVF", SAMPLE + "_1.jpg"))
    radar = Image.open(os.path.join(ROOT, "LRVF", SAMPLE + "_2.jpg"))

    print("inputs:", image.size, lidar.size, radar.size)
    result = yolo.detect_image(image, radar_image=radar, lidar_image=lidar)

    out = os.path.join("logs", "predict_%s.jpg" % SAMPLE)
    result.save(out)
    print("prediction saved to", out, "size:", result.size)


if __name__ == "__main__":
    main()
