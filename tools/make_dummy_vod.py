"""Create a synthetic mini "VoD-like" dataset so vod_to_voc.py can be tested
without downloading the real View-of-Delft release.

Produces:
    <root>/lidar/training/image_2/<id>.jpg
    <root>/lidar/training/velodyne/<id>.bin            Nx4 float32
    <root>/lidar/training/radar_3_scans/<id>.bin       Nx7 float32 (time in 0,-1,-2)
    <root>/lidar/training/label_2/<id>.txt             KITTI label lines
    <root>/lidar/training/calib/<id>.txt               KITTI calibration stub

Usage:
    python tools/make_dummy_vod.py --n 4 --out /tmp/dummy_vod
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Car", "Pedestrian", "Cyclist", "bicycle_rack", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="dummy_vod")
    ap.add_argument("--width", type=int, default=1936)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--points", type=int, default=2000)
    a = ap.parse_args()

    root = os.path.join(a.out, "lidar", "training")
    for sub in ("image_2", "velodyne", "radar_3_scans", "label_2", "calib"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(a.n):
        image_id = "%06d" % i

        img = Image.new("RGB", (a.width, a.height), (60, 70, 80))
        draw = ImageDraw.Draw(img)
        lines = []
        for k in range(3):
            w = int(rng.integers(80, 300))
            h = int(rng.integers(80, 300))
            x = int(rng.integers(0, a.width - w - 1))
            y = int(rng.integers(0, a.height - h - 1))
            draw.rectangle([x, y, x + w, y + h], outline=(220, 220, 220), width=3)
            cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
            lines.append(
                "%s 0.0 0 -0.5 %.2f %.2f %.2f %.2f 1.60 0.60 2.00 %.2f %.2f 10.0 %.3f"
                % (cls, x, y, x + w, y + h, float(rng.uniform(-5, 5)),
                   float(rng.uniform(-5, 5)), float(rng.uniform(-1.5, 1.5)))
            )
        img.save(os.path.join(root, "image_2", image_id + ".jpg"), quality=90)

        # LiDAR: Nx4
        lidar = np.zeros((a.points, 4), dtype=np.float32)
        lidar[:, 0] = rng.uniform(0.5, 40.0, a.points)
        lidar[:, 1] = rng.uniform(-20.0, 20.0, a.points)
        lidar[:, 2] = rng.uniform(-2.0, 0.5, a.points)
        lidar[:, 3] = rng.uniform(0, 1, a.points)
        lidar.tofile(os.path.join(root, "velodyne", image_id + ".bin"))

        # radar: Nx7 with time in {0,-1,-2}
        n_r = a.points // 4
        radar = np.zeros((n_r * 3, 7), dtype=np.float32)
        tvals = np.repeat(np.array([0.0, -1.0, -2.0], dtype=np.float32), n_r)
        radar[:, 0] = rng.uniform(0.5, 40.0, n_r * 3)
        radar[:, 1] = rng.uniform(-20.0, 20.0, n_r * 3)
        radar[:, 2] = rng.uniform(-2.0, 0.5, n_r * 3)
        radar[:, 3] = rng.uniform(0, 30, n_r * 3)          # RCS
        radar[:, 4] = rng.uniform(-10, 10, n_r * 3)        # v_r
        radar[:, 5] = rng.uniform(-10, 10, n_r * 3)        # v_r compensated
        radar[:, 6] = tvals
        radar.tofile(os.path.join(root, "radar_3_scans", image_id + ".bin"))

        with open(os.path.join(root, "label_2", image_id + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        with open(os.path.join(root, "calib", image_id + ".txt"), "w") as f:
            f.write("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
            f.write("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n")
            f.write("Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
            f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")

    print("generated %d fake VoD frames under %s" % (a.n, a.out))


if __name__ == "__main__":
    main()
