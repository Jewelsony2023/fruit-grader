"""
Convert a COCO-format polygon dataset (Roboflow / CVAT export) into class-id
masks and add it to labels.csv as a VERIFIED test set.

Use this for:
  * the Roboflow/HF fruit segmentation set (131 images) from the deck,
  * masks your team hand-corrects in CVAT (export as "COCO 1.0").

Category names in the json are mapped to our classes with --map, e.g.
    --map "apple=healthy,fruit=healthy,bruise=bruise,rotten=rot,rot=rot,cut=mechanical"
Healthy/fruit polygons are drawn first, defect polygons on top.

Usage:
    python -m fruitgrade.data.coco_to_masks --json ann.json --images imgs/ --map "..."
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
from fruitgrade.data.synthetic import write_labels
from fruitgrade.grading import grade_from_severity, severity_from_mask

CLASS_IDS = {"background": 0, "healthy": 1, "bruise": 2, "rot": 3, "mechanical": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--map", required=True, help='"cocoName=ourClass,..."')
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fruit", default="unknown")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["data"]["root"])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    name_map = {}
    for pair in args.map.split(","):
        k, v = pair.split("=")
        name_map[k.strip().lower()] = CLASS_IDS[v.strip().lower()]

    coco = json.load(open(args.json, encoding="utf-8"))
    cat = {c["id"]: c["name"].lower() for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    rows, skipped = [], Counter()
    for im in coco["images"]:
        bgr = cv2.imread(str(Path(args.images) / im["file_name"]))
        if bgr is None:
            skipped["missing image"] += 1
            continue
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        items = []
        for a in anns[im["id"]]:
            cid = name_map.get(cat[a["category_id"]])
            if cid is None:
                skipped[f"unmapped:{cat[a['category_id']]}"] += 1
                continue
            items.append((cid, a))
        for cid, a in sorted(items, key=lambda t: t[0]):   # healthy first, defects on top
            seg = a.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], cid)
            else:
                skipped["RLE not supported"] += 1
        sev = severity_from_mask(mask)
        g = grade_from_severity(sev, cfg["grading_rules"])
        stem = f"coco_{Path(im['file_name']).stem}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), bgr)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        rows.append({"image": f"images/{stem}.jpg", "mask": f"masks/{stem}.png",
                     "grade": cfg["grades"][g], "severity": round(sev["total_pct"], 3),
                     "split": args.split, "fruit": args.fruit, "source": "coco",
                     "mask_quality": "verified"})

    write_labels(out / "labels.csv", rows, append=True)
    print(f"Added {len(rows)} verified samples. Skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
