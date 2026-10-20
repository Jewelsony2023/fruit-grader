"""
Export the test originals for hand-correction in CVAT (Phase 3).

For every test row of labels.csv (pseudo-masks, 227 images) this writes the image
under a NEUTRAL name (t001.jpg ...; the processed names contain "fr"/"rot", which
would bias annotators) and COCO 1.0 polygon pre-annotations made from its pseudo-mask:
    fruit  : the whole fruit outline (draw/keep it around the WHOLE fruit, defects included)
    bruise, rot, cut : one polygon per defect region (cut = our "mechanical")
    healthy: healthy skin enclosed by a defect outline (a polygon cannot have holes,
             so these islands are drawn back on top of the defects)
Images are split into --batches batches, stratified by class (fresh/rotten x fruit),
one batch per annotator.

Output (default data/cvat/):
    batch_<k>/images/t###.jpg
    batch_<k>/annotations/instances_default.json   COCO 1.0 pre-annotations
    batch_<k>_images.zip                           upload when creating the CVAT task
    batch_<k>_annotations.zip                      upload as "COCO 1.0" annotations
    manifest.csv                                   t###.jpg -> processed image, src, group, class
After annotation, convert each exported batch back with coco_to_masks --manifest.

Usage:
    python -m fruitgrade.data.export_for_cvat --root data/processed --out data/cvat
"""
import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.data.dataset import read_labels
from fruitgrade.grading import BRUISE, HEALTHY, MECHANICAL, ROT

CATEGORIES = [{"id": 1, "name": "fruit", "supercategory": ""},
              {"id": 2, "name": "bruise", "supercategory": ""},
              {"id": 3, "name": "rot", "supercategory": ""},
              {"id": 4, "name": "cut", "supercategory": ""},
              {"id": 5, "name": "healthy", "supercategory": ""}]
# category id -> which mask pixels it covers
CAT_PIXELS = {1: lambda m: m > 0, 2: lambda m: m == BRUISE, 3: lambda m: m == ROT, 4: lambda m: m == MECHANICAL,
              5: lambda m: healthy_islands(m)}
# for coco_to_masks: CVAT label name -> our class ("healthy_patch" is drawn last)
CVAT_MAP = "fruit=healthy,bruise=bruise,rot=rot,cut=mechanical,healthy=healthy_patch"


def healthy_islands(mask):
    """Healthy pixels enclosed by a defect region's outer contour."""
    defect = (mask >= BRUISE).astype(np.uint8)
    cnts, _ = cv2.findContours(defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(defect)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return (filled > 0) & (mask == HEALTHY)


def mask_to_polygons(binary, eps=1.0, min_area=4):
    """Outer contours of a bool mask as flat COCO polygons [x1, y1, x2, y2, ...]."""
    cnts, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(float).flatten().tolist())
    return polys


def coco_for(images, eps=1.0):
    """images: list of (file_name, mask). Returns a COCO dict with one annotation per polygon."""
    coco = {"info": {"description": "fruit-grader pseudo-mask pre-annotations"}, "licenses": [],
            "categories": CATEGORIES, "images": [], "annotations": []}
    ann_id = 1
    for img_id, (name, mask) in enumerate(images, 1):
        h, w = mask.shape
        coco["images"].append({"id": img_id, "file_name": name, "width": w, "height": h})
        for cat_id, sel in CAT_PIXELS.items():
            for poly in mask_to_polygons(sel(mask), eps):
                pts = np.array(poly, np.float32).reshape(-1, 2)
                x, y = pts.min(0)
                bw, bh = pts.max(0) - pts.min(0)
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id, "segmentation": [poly],
                    "area": float(cv2.contourArea(pts)), "bbox": [float(x), float(y), float(bw), float(bh)],
                    "iscrowd": 0})
                ann_id += 1
    return coco


def assign_batches(rows, n_batches, seed):
    """Stratified by class: shuffle within each class, then deal all classes round-robin."""
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    order = []
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda r: r["image"])
        order += [items[i] for i in rng.permutation(len(items))]
    return {r["image"]: i % n_batches + 1 for i, r in enumerate(order)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--out", default="data/cvat")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1.5, help="polygon simplification in pixels")
    ap.add_argument("--limit", type=int, default=0, help="only the first N test images (round-trip check)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    rows = [r for r in read_labels(root, "test") if r["mask_quality"] == "pseudo"]
    rows = sorted(rows, key=lambda r: r["image"])
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        r["cls"] = r["source"].replace("kaggle_", "") + "_" + r["fruit"]
    batch_of = assign_batches(rows, args.batches, args.seed)

    per_batch, manifest = defaultdict(list), []
    for i, r in enumerate(rows, 1):
        name, b = f"t{i:03d}.jpg", batch_of[r["image"]]
        bdir = out / f"batch_{b}"
        (bdir / "images").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / r["image"], bdir / "images" / name)
        mask = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
        per_batch[b].append((name, mask))
        manifest.append({"file_name": name, "batch": b, "image": r["image"], "mask": r["mask"],
                         "src": r["src"], "group": r["group"], "cls": r["cls"], "fruit": r["fruit"],
                         "pseudo_grade": r["grade"]})

    for b, items in sorted(per_batch.items()):
        bdir = out / f"batch_{b}"
        (bdir / "annotations").mkdir(parents=True, exist_ok=True)
        ann = bdir / "annotations" / "instances_default.json"
        ann.write_text(json.dumps(coco_for(items, args.eps)))
        with zipfile.ZipFile(out / f"batch_{b}_annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(ann, "annotations/instances_default.json")
        with zipfile.ZipFile(out / f"batch_{b}_images.zip", "w", zipfile.ZIP_STORED) as z:
            for name, _ in items:
                z.write(bdir / "images" / name, name)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"{len(rows)} test images -> {out}")
    for b in sorted(per_batch):
        print(f"  batch_{b}: {len(per_batch[b])} images",
              dict(sorted(Counter(m["cls"] for m in manifest if m["batch"] == b).items())))


if __name__ == "__main__":
    main()
