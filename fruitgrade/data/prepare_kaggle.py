"""
Turn a fresh/rotten folder dataset (e.g. Kaggle "Fruits fresh and rotten for
classification") into this project's format:  images/ + masks/ + labels.csv

Which images and which split come ONLY from the splits file written by
fruitgrade.data.dedupe (originals only, split by near-duplicate group).
Folder names give fresh/rotten and the fruit, e.g. train/rottenbanana/X.png.
There is no random splitting here: a near-copy in another split would leak.

Grade labels are DERIVED from the pseudo-mask severity with the rules in
config.yaml (the Kaggle data only says fresh/rotten — it has no A/B/C grades).

Usage:
    python -m fruitgrade.data.prepare_kaggle --src D:/datasets/fruits/dataset --out data/processed
    # second round, after training a model (Grad-CAM refinement):
    python -m fruitgrade.data.prepare_kaggle --src ... --cam-checkpoint runs/baseline/best.pt
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
from fruitgrade.data.label_filter import apply_filter
from fruitgrade.data.pseudo_masks import pseudo_mask
from fruitgrade.data.synthetic import write_labels
from fruitgrade.grading import colourise_mask, grade_from_severity, severity_from_mask

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRUIT_WORDS = ["apple", "banana", "orange", "mango", "tomato", "potato", "cucumber",
               "pepper", "capsicum", "guava", "pomegranate", "lime", "strawberr",
               "bitter", "okra", "grape", "pear"]


def parse_path(p: Path):
    s = "/".join(x.lower() for x in p.parts)
    rotten = "rotten" in s or "stale" in s
    fresh = "fresh" in s
    if not (rotten or fresh):
        return None
    fruit = next((f for f in FRUIT_WORDS if f in s), "unknown")
    is_test = re.search(r"(^|/)test(/|$)", s) is not None
    return {"rotten": rotten, "fruit": fruit, "is_test": is_test}


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def load_splits(path):
    """Rows of the dedupe splits file: path, cls, group, split, ..."""
    return list(csv.DictReader(open(path, encoding="utf-8")))


def check_no_group_leak(rows):
    """Raise if any near-duplicate group appears in more than one split."""
    splits = {}
    for r in rows:
        splits.setdefault(r["group"], set()).add(r["split"])
    leaks = {g: sorted(s) for g, s in splits.items() if len(s) > 1}
    if leaks:
        raise ValueError(f"{len(leaks)} groups span several splits, e.g. {list(leaks.items())[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset tree the splits-file paths are relative to")
    ap.add_argument("--splits", default="data/interim/splits.csv", help="output of fruitgrade.data.dedupe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--limit-per-class", type=int, default=0, help="0 = use all")
    ap.add_argument("--preview", type=int, default=40, help="save N overlay previews to check by eye")
    ap.add_argument("--cam-checkpoint", default=None, help="optional: refine masks with Grad-CAM")
    ap.add_argument("--append", action="store_true", help="append to existing labels.csv")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["data"]["root"])
    for d in ("images", "masks", "preview"):
        (out / d).mkdir(parents=True, exist_ok=True)

    cam_fn = None
    if args.cam_checkpoint:
        from fruitgrade.inference import Predictor
        pred = Predictor(args.cam_checkpoint)
        reject = pred.grades.index("Reject")
        cam_fn = lambda rgb: pred.gradcam(rgb, target=reject)  # noqa: E731

    entries = load_splits(args.splits)
    check_no_group_leak(entries)
    per_class = Counter()
    rows, skipped, n_prev = [], [], 0
    for i, e in enumerate(entries):
        p = Path(args.src) / e["path"]
        info = parse_path(Path(e["path"]))
        key = (info["fruit"], info["rotten"])
        if args.limit_per_class and per_class[key] >= args.limit_per_class:
            continue
        per_class[key] += 1

        bgr = cv2.imread(str(p))
        if bgr is None:
            skipped.append({"src": e["path"], "reason": "unreadable"})
            continue
        rgb = cv2.cvtColor(resize_max(bgr, args.max_side), cv2.COLOR_BGR2RGB)
        cam = None
        if cam_fn is not None and info["rotten"]:
            cam = cv2.resize(cam_fn(rgb), rgb.shape[1::-1])
        mask = pseudo_mask(rgb, is_rotten=info["rotten"], cam=cam)
        sev = severity_from_mask(mask)
        if sev["fruit_pixels"] < 0.05 * mask.size:   # fruit not found -> skip (and log it)
            skipped.append({"src": e["path"], "reason": "fruit not found"})
            continue
        g = grade_from_severity(sev, cfg["grading_rules"])

        name = f"kg_{'rot' if info['rotten'] else 'fr'}_{info['fruit']}_{i:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)
        rows.append({"image": f"images/{name}.jpg", "mask": f"masks/{name}.png",
                     "grade": cfg["grades"][g], "severity": round(sev["total_pct"], 3),
                     "split": e["split"], "fruit": info["fruit"],
                     "source": "kaggle_rotten" if info["rotten"] else "kaggle_fresh",
                     "mask_quality": "pseudo", "group": e["group"], "src": e["path"]})

        if n_prev < args.preview:
            over = rgb.copy()
            col = colourise_mask(mask)
            sel = mask >= 2
            over[sel] = (0.45 * over[sel] + 0.55 * col[sel]).astype(np.uint8)
            tile = np.hstack([rgb, over])
            cv2.imwrite(str(out / "preview" / f"{name}.jpg"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            n_prev += 1
        if len(rows) % 200 == 0:
            print(f"  processed {len(rows)} images ...")

    check_no_group_leak(rows)
    apply_filter(rows)
    write_labels(out / "labels.csv", rows, append=args.append)
    with open(out / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src", "reason"])
        w.writeheader()
        w.writerows(skipped)
    print(f"Wrote {len(rows)} samples -> {out}  (skipped {len(skipped)}, see skipped.csv)")
    print("grades :", Counter(r["grade"] for r in rows))
    print("splits :", Counter(r["split"] for r in rows))
    print(f"Check previews in {out / 'preview'} before training!")


if __name__ == "__main__":
    main()
