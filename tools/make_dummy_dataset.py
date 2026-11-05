"""Generate a tiny synthetic dataset in the layout expected by tools/build_annotations.py.

Purpose: smoke-test the full pipeline (data loader -> model -> loss -> checkpoint)
without the real VoD/Flow data.

Layout produced under VOCdevkit/VOC2007:
    JPEGImages/<id>.jpg      RGB image
    LRVF/<id>_1.jpg          BEV frame t
    LRVF/<id>_2.jpg          BEV frame t-1
    LRVF/<id>_3.jpg          BEV frame t-2
    Annotations/<id>.xml     VOC XML
    ImageSets/Main/{train,val,test}.txt

Usage:
    python tools/make_dummy_dataset.py --n 16 --seed 0
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

SIZE = 640


def make_rgb(rng, path):
    arr = np.random.randint(40, 90, (SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    boxes = []
    for _ in range(rng.randint(1, 3)):
        w = rng.randint(60, 200)
        h = rng.randint(60, 200)
        x = rng.randint(0, SIZE - w - 1)
        y = rng.randint(0, SIZE - h - 1)
        color = tuple(int(c) for c in np.random.randint(120, 255, 3))
        draw.rectangle([x, y, x + w, y + h], fill=color)
        boxes.append((x, y, x + w, y + h))
    img.save(path, quality=95)
    return boxes


def make_bev(rng, path):
    arr = np.random.randint(0, 40, (SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(20, 60)):
        cx = rng.randint(0, SIZE - 1)
        cy = rng.randint(0, SIZE - 1)
        v = rng.randint(120, 255)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(v, v, v))
    img.save(path, quality=95)


def write_xml(path, filename, boxes, classes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2007"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(SIZE)
    ET.SubElement(size, "height").text = str(SIZE)
    ET.SubElement(size, "depth").text = "3"
    for (x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = random.choice(classes)
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default="model_data/classes.txt")
    ap.add_argument("--root", default="VOCdevkit/VOC2007")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    np.random.seed(a.seed)

    classes = [c.strip() for c in open(a.classes, encoding="utf-8") if c.strip()]

    dirs = {
        "jpg": os.path.join(a.root, "JPEGImages"),
        "bev": os.path.join(a.root, "LRVF"),
        "xml": os.path.join(a.root, "Annotations"),
        "sets": os.path.join(a.root, "ImageSets", "Main"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    ids = []
    for i in range(a.n):
        image_id = "%06d" % i
        ids.append(image_id)
        boxes = make_rgb(rng, os.path.join(dirs["jpg"], image_id + ".jpg"))
        for k in (1, 2):
            make_bev(rng, os.path.join(dirs["bev"], "%s_%d.jpg" % (image_id, k)))
        write_xml(os.path.join(dirs["xml"], image_id + ".xml"), image_id + ".jpg", boxes, classes)

    n_train = max(1, int(a.n * 0.7))
    n_val = max(1, int(a.n * 0.15))
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    for name, lst in splits.items():
        if not lst:
            lst = ids[-1:]
        with open(os.path.join(dirs["sets"], name + ".txt"), "w") as f:
            f.write("\n".join(lst) + "\n")

    print("generated %d samples under %s" % (a.n, a.root))
    print("splits:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
