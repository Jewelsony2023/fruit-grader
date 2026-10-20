"""
Turn a fresh/rotten folder dataset (e.g. Kaggle "Fruits fresh and rotten for
classification") into this project's format:  images/ + masks/ + labels.csv

Expected input: any folder tree where the folder names contain "fresh" or
"rotten" and (optionally) the fruit name, e.g.
    dataset/train/freshapples/*.png
    dataset/train/rottenbanana/*.png
    dataset/test/freshoranges/*.png
If the path contains a "test" folder those images go to the test split;
the rest are split train/val.

Grade labels are DERIVED from the pseudo-mask severity with the rules in
config.yaml (the Kaggle data only says fresh/rotten — it has no A/B/C grades).

Usage:
    python -m fruitgrade.data.prepare_kaggle --src path/to/dataset --out data/processed
    # second round, after training a model (Grad-CAM refinement):
    python -m fruitgrade.data.prepare_kaggle --src ... --cam-checkpoint runs/baseline/best.pt
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--limit-per-class", type=int, default=0, help="0 = use all")
    ap.add_argument("--val-frac", type=float, default=0.15)
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

    files = [p for p in Path(args.src).rglob("*") if p.suffix.lower() in IMG_EXT]
    rng = np.random.default_rng(cfg["train"]["seed"])
    rng.shuffle(files)
    per_class = Counter()
    rows, n_prev = [], 0
    for i, p in enumerate(files):
        info = parse_path(p.relative_to(args.src))
        if info is None:
            continue
        key = (info["fruit"], info["rotten"])
        if args.limit_per_class and per_class[key] >= args.limit_per_class:
            continue
        per_class[key] += 1

        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(resize_max(bgr, args.max_side), cv2.COLOR_BGR2RGB)
        cam = None
        if cam_fn is not None and info["rotten"]:
            cam = cv2.resize(cam_fn(rgb), rgb.shape[1::-1])
        mask = pseudo_mask(rgb, is_rotten=info["rotten"], cam=cam)
        sev = severity_from_mask(mask)
        if sev["fruit_pixels"] < 0.05 * mask.size:   # fruit not found -> skip
            continue
        g = grade_from_severity(sev, cfg["grading_rules"])

        name = f"kg_{'rot' if info['rotten'] else 'fr'}_{info['fruit']}_{i:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)
        split = "test" if info["is_test"] else ("val" if rng.random() < args.val_frac else "train")
        rows.append({"image": f"images/{name}.jpg", "mask": f"masks/{name}.png",
                     "grade": cfg["grades"][g], "severity": round(sev["total_pct"], 3),
                     "split": split, "fruit": info["fruit"],
                     "source": "kaggle_rotten" if info["rotten"] else "kaggle_fresh",
                     "mask_quality": "pseudo"})

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

    write_labels(out / "labels.csv", rows, append=args.append)
    print(f"Wrote {len(rows)} samples -> {out}")
    print("grades :", Counter(r["grade"] for r in rows))
    print("splits :", Counter(r["split"] for r in rows))
    print(f"Check previews in {out / 'preview'} before training!")


if __name__ == "__main__":
    main()
