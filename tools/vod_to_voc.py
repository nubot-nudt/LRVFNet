"""Convert the View-of-Delft (VoD) dataset into the VOC + LRVF layout used here.

Input  (VoD, KITTI-style):
    <vod>/lidar/training/image_2/<id>.jpg
    <vod>/lidar/training/velodyne/<id>.bin            Nx4  [x, y, z, reflectance]
    <vod>/lidar/training/radar_3_scans/<id>.bin       Nx7  [x, y, z, RCS, v_r, v_r_c, time]
    <vod>/lidar/training/label_2/<id>.txt             KITTI labels (2D boxes pre-projected)
    <vod>/lidar/training/calib/<id>.txt               KITTI calibration

Output (what tools/build_annotations.py expects):
    <out>/JPEGImages/<id>.jpg          RGB
    <out>/LRVF/<id>_1.jpg              BEV frame t   (time ==  0)
    <out>/LRVF/<id>_2.jpg              BEV frame t-1 (time == -1)
    <out>/LRVF/<id>_3.jpg              BEV frame t-2 (time == -2)
    <out>/Annotations/<id>.xml         VOC XML
    <out>/ImageSets/Main/{train,val,test}.txt

BEV generation follows paper section 3.1: project points onto a 0.1 m grid over
X in [0,40], Y in [-20,20], keep the maximum height per cell, normalise to 0-255.

Usage:
    python tools/vod_to_voc.py --vod /path/to/vod --out VOCdevkit/VOC2007
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

DEFAULT_CLASSES = [
    "bicycle_rack", "Car", "Pedestrian", "bicycle", "Cyclist", "rider",
    "moped_scooter", "ride_other", "truck", "motor", "ride_uncertain",
    "vehicle_other", "human_depiction",
]


def read_points(path, cols):
    if not os.path.exists(path):
        return np.zeros((0, cols), dtype=np.float32)
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % cols:
        pts = pts[: pts.size - pts.size % cols]
    return pts.reshape(-1, cols)


def bev_from_points(pts, cfg, out_size):
    """pts: NxC with columns [x, y, z, ...] in metres, vehicle frame."""
    x_min, x_max, y_min, y_max = cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"]
    res = cfg["res"]
    w = int(round((x_max - x_min) / res))
    h = int(round((y_max - y_min) / res))
    img = np.zeros((h, w), dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)

    if pts.shape[0]:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        m = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        x, y, z = x[m], y[m], z[m]
        # row 0 = farthest forward, col 0 = leftmost; flip vertically for image view
        ci = ((y - y_min) / res).astype(np.int32)
        ri = h - 1 - ((x - x_min) / res).astype(np.int32)
        ri = np.clip(ri, 0, h - 1)
        ci = np.clip(ci, 0, w - 1)
        for r, c, vz in zip(ri, ci, z):
            if (not filled[r, c]) or vz > img[r, c]:
                img[r, c] = vz
                filled[r, c] = True

    # Empty cells stay black; occupied cells encode height as 0-255 grey.
    z0, z1 = cfg["z_min"], cfg["z_max"]
    gray = np.zeros((h, w), dtype=np.float32)
    if filled.any():
        gray[filled] = np.clip((img[filled] - z0) / (z1 - z0), 0.0, 1.0) * 255.0
    gray = gray.astype(np.uint8)
    im = Image.fromarray(gray, "L").convert("RGB")
    if im.size != (out_size, out_size):
        im = im.resize((out_size, out_size), Image.BILINEAR)
    return im


def parse_kitti_labels(path, classes):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in classes:
                continue
            occ = int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0
            if occ >= 2:
                continue
            x1, y1, x2, y2 = (float(v) for v in parts[4:8])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((cls, int(x1), int(y1), int(x2), int(y2)))
    return boxes


def write_xml(path, filename, boxes, width, height):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2007"
    ET.SubElement(root, "filename").text = filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = "ViewOfDelft"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for (cls, x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(max(0, x1))
        ET.SubElement(bnd, "ymin").text = str(max(0, y1))
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vod", required=True, help="root of the VoD release (contains lidar/training)")
    ap.add_argument("--out", default="VOCdevkit/VOC2007")
    ap.add_argument("--size", type=int, default=640, help="BEV output size in pixels")
    ap.add_argument("--res", type=float, default=0.1, help="BEV cell size in metres")
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=40.0)
    ap.add_argument("--y-min", type=float, default=-20.0)
    ap.add_argument("--y-max", type=float, default=20.0)
    ap.add_argument("--z-min", type=float, default=-2.0)
    ap.add_argument("--z-max", type=float, default=0.5)
    ap.add_argument("--split", default="0.7,0.15,0.15", help="train,val,test fractions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = dict(x_min=a.x_min, x_max=a.x_max, y_min=a.y_min, y_max=a.y_max,
               z_min=a.z_min, z_max=a.z_max, res=a.res)
    classes = DEFAULT_CLASSES

    root = os.path.join(a.vod, "lidar", "training")
    dirs = {k: os.path.join(root, k) for k in ("image_2", "velodyne", "radar_3_scans", "label_2")}
    for k, d in dirs.items():
        if not os.path.isdir(d):
            raise SystemExit("missing %s (expected %s)" % (k, d))

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(dirs["image_2"]) if f.endswith(".jpg"))
    if not ids:
        raise SystemExit("no frames found in %s" % dirs["image_2"])

    out_jpg = os.path.join(a.out, "JPEGImages")
    out_bev = os.path.join(a.out, "LRVF")
    out_xml = os.path.join(a.out, "Annotations")
    out_sets = os.path.join(a.out, "ImageSets", "Main")
    for d in (out_jpg, out_bev, out_xml, out_sets):
        os.makedirs(d, exist_ok=True)

    for image_id in ids:
        img = Image.open(os.path.join(dirs["image_2"], image_id + ".jpg")).convert("RGB")
        img.save(os.path.join(out_jpg, image_id + ".jpg"), quality=95)
        wdt, hgt = img.size

        # Paper Fig.2: two BEV inputs (LiDAR, mmWave radar).  _1 = LiDAR, _2 = radar.
        lidar = read_points(os.path.join(dirs["velodyne"], image_id + ".bin"), 4)
        rad = read_points(os.path.join(dirs["radar_3_scans"], image_id + ".bin"), 7)
        radar = rad[np.isclose(rad[:, 6], 0)] if rad.shape[0] else rad
        frames = [lidar, radar]

        for k, fp in enumerate(frames, start=1):
            bev_from_points(fp, cfg, a.size).save(
                os.path.join(out_bev, "%s_%d.jpg" % (image_id, k)), quality=95)

        boxes = parse_kitti_labels(os.path.join(dirs["label_2"], image_id + ".txt"), classes)
        write_xml(os.path.join(out_xml, image_id + ".xml"), image_id + ".jpg", boxes, wdt, hgt)

    tr, va, te = (float(x) for x in a.split.split(","))
    random.Random(a.seed).shuffle(ids)
    n_tr = int(len(ids) * tr)
    n_va = int(len(ids) * va)
    splits = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}
    for name, lst in splits.items():
        with open(os.path.join(out_sets, name + ".txt"), "w") as f:
            f.write("\n".join(lst) + ("\n" if lst else ""))

    print("converted %d frames -> %s" % (len(ids), a.out))
    print("splits:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
