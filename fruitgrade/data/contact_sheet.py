"""
Contact sheets to check pseudo-masks by eye (Phase 2).

For N random images per class (fresh/rotten x fruit), one tile shows:
    input | overlay (fruit outline in cyan, defects filled) | severity per type + grade
Defect colours: bruise purple, rot red, mechanical amber.

The sample is fixed by --seed, so "before" and "after" sheets show the same images.
With --recompute the masks are rebuilt from the ORIGINAL files with the current
pseudo_masks.py (same input as prepare_kaggle), so a tuning change can be
compared without re-running the whole dataset.

Usage:
    python -m fruitgrade.data.contact_sheet --root data/processed --out data/interim/sheets_before
    python -m fruitgrade.data.contact_sheet --root data/processed --out data/interim/sheets_after \
        --recompute --src D:/datasets/fruits/dataset
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
from fruitgrade.data.dataset import read_labels
from fruitgrade.data.prepare_kaggle import resize_max
from fruitgrade.data.pseudo_masks import pseudo_mask
from fruitgrade.grading import HEALTHY, colourise_mask, grade_from_severity, severity_from_mask

TILE = 170


def sample_rows(rows, per_class, seed):
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[f"{r['source'].replace('kaggle_', '')}_{r['fruit']}"].append(r)
    rng = np.random.default_rng(seed)
    out = {}
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda r: r["image"])
        idx = rng.choice(len(items), min(per_class, len(items)), replace=False)
        out[cls] = [items[i] for i in sorted(idx)]
    return out


def overlay(rgb, mask):
    over = rgb.copy()
    col = colourise_mask(mask)
    sel = mask > HEALTHY
    over[sel] = (0.35 * over[sel] + 0.65 * col[sel]).astype(np.uint8)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(over, cnts, -1, (0, 200, 255), max(1, rgb.shape[0] // 150))
    return over


def tile(num, rgb, mask, grades, rules):
    sev = severity_from_mask(mask)
    g = grades[grade_from_severity(sev, rules, grades)]
    sq = lambda im: cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_AREA)  # noqa: E731
    panel = np.full((TILE, 118, 3), 255, np.uint8)
    lines = [(f"#{num}", (0, 0, 0)), (f"grade {g}", (0, 0, 0)),
             (f"total {sev['total_pct']:.1f}%", (0, 0, 0)),
             (f"bruise {sev['bruise_pct']:.1f}%", (155, 89, 182)),
             (f"rot {sev['rot_pct']:.1f}%", (200, 40, 40)),
             (f"mech {sev['mechanical_pct']:.1f}%", (190, 130, 0))]
    for k, (txt, c) in enumerate(lines):
        cv2.putText(panel, txt, (6, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if k < 2 else 0.47, c, 1 + (k < 2), cv2.LINE_AA)
    t = np.hstack([sq(rgb), np.full((TILE, 3, 3), 255, np.uint8), sq(overlay(rgb, mask)), panel])
    return np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=225), sev, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="rebuild masks with the current pseudo_masks.py")
    ap.add_argument("--src", default=None, help="original dataset tree (needed with --recompute)")
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = sample_rows(read_labels(root), args.per_class, args.seed)
    log = []
    for cls, rows in picks.items():
        tiles = []
        for num, r in enumerate(rows, 1):
            if args.recompute:
                bgr = resize_max(cv2.imread(str(Path(args.src) / r["src"])), args.max_side)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = pseudo_mask(rgb, is_rotten=r["source"] == "kaggle_rotten")
            else:
                rgb = cv2.cvtColor(cv2.imread(str(root / r["image"])), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
            t, sev, g = tile(num, rgb, mask, cfg["grades"], cfg["grading_rules"])
            tiles.append(t)
            log.append({"class": cls, "num": num, "image": r["image"], "src": r["src"], "grade": g,
                        **{k: round(sev[k], 2) for k in ("total_pct", "bruise_pct", "rot_pct", "mechanical_pct")}})
        while len(tiles) % args.cols:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)])
        head = np.full((34, sheet.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{cls}  ({len(rows)} random, seed {args.seed})   input | overlay: outline=fruit, "
                    "purple=bruise, red=rot, amber=mech", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.imwrite(str(out / f"{cls}.jpg"), cv2.cvtColor(np.vstack([head, sheet]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(out / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"{len(picks)} sheets -> {out}")


if __name__ == "__main__":
    main()
