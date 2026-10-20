"""
Pseudo-mask tuning check: two numbers, used while tuning pseudo_masks.py.

1. Fresh/rotten agreement (real Kaggle photos, no ground-truth masks):
   40 random images per class (seed 1) get pseudo-masks and a grade. An image
   "agrees" if a fresh-labelled photo grades A or B, or a rotten-labelled photo
   grades C or Reject. This is a sanity proxy, NOT accuracy: a fresh fruit can
   honestly be a B/C. A fresh photo graded Reject, though, is clearly wrong.
2. Synthetic exact-mask accuracy (data/synthetic, exact masks known):
   defect IoU (any defect vs not, pooled over pixels), severity MAE in % points,
   and fruit IoU (fruit vs background, mean per image).

Note: the agreement sample is drawn from ALL splits (train, val and test), so it
is not a held-out measurement; see "pseudo_mask_tuning_sample" in dataset_audit.json.

Usage:
    python -m fruitgrade.data.pm_eval
    python -m fruitgrade.data.pm_eval --audit data/interim/dataset_audit.json   # record sample splits
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
from fruitgrade.data.prepare_kaggle import resize_max
from fruitgrade.data.pseudo_masks import pseudo_mask
from fruitgrade.grading import grade_from_severity, severity_from_mask

CFG = load_config()
G = CFG["grades"]


def real_one(r, src):
    bgr = resize_max(cv2.imread(str(Path(src) / r["src"])), 384)
    m = pseudo_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), is_rotten=r["source"] == "kaggle_rotten")
    s = severity_from_mask(m)
    return r["cls"], G[grade_from_severity(s, CFG["grading_rules"], G)], s["mechanical_pct"]


def synth_one(r, root):
    img = cv2.cvtColor(cv2.imread(str(Path(root) / r["image"])), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(Path(root) / r["mask"]), cv2.IMREAD_UNCHANGED)
    pm = pseudo_mask(img, is_rotten=float(r["severity"]) > 0)
    gd, pd = gt >= 2, pm >= 2
    return (int((gd & pd).sum()), int((gd | pd).sum()),
            abs(severity_from_mask(gt)["total_pct"] - severity_from_mask(pm)["total_pct"]),
            ((gt > 0) & (pm > 0)).sum() / max(((gt > 0) | (pm > 0)).sum(), 1))


def read_rows(root):
    """All labels.csv rows, excluded ones included (the samples were drawn before the filter)."""
    return list(csv.DictReader(open(Path(root) / "labels.csv", encoding="utf-8")))


def tuning_sample(labels, per_class=40, seed=1):
    by = defaultdict(list)
    for r in labels:
        r["cls"] = r["source"].replace("kaggle_", "") + "_" + r["fruit"]
        by[r["cls"]].append(r)
    rng = np.random.default_rng(seed)
    return [by[c][i] for c in sorted(by) for i in rng.choice(len(by[c]), per_class, replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--src", default="D:/datasets/fruits/dataset")
    ap.add_argument("--synthetic", default="data/synthetic")
    ap.add_argument("--n-synthetic", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--audit", default=None, help="only record the sample's split counts in this audit json")
    args = ap.parse_args()

    pick = tuning_sample(read_rows(args.root))
    if args.audit:
        p = Path(args.audit)
        audit = json.loads(p.read_text())
        audit["pseudo_mask_tuning_sample"] = {
            "what": "40 random images per class (seed 1) used to tune pseudo_masks.py (fresh/rotten agreement)",
            "images": len(pick), "by_split": dict(Counter(r["split"] for r in pick)),
            "note": "drawn from all splits: val/test images influenced pseudo-mask tuning; not fixed, reported"}
        from fruitgrade.data.contact_sheet import sample_rows
        sheets = [r for rs in sample_rows(read_rows(args.root), 24, 0).values() for r in rs]
        audit["pseudo_mask_contact_sheet_sample"] = {
            "what": "24 random images per class (seed 0) shown on the before/after review sheets",
            "images": len(sheets), "by_split": dict(Counter(r["split"] for r in sheets))}
        p.write_text(json.dumps(audit, indent=2))
        print({k: audit[k] for k in ("pseudo_mask_tuning_sample", "pseudo_mask_contact_sheet_sample")})
        return

    srows = [r for r in csv.DictReader(open(Path(args.synthetic) / "labels.csv", encoding="utf-8"))
             if r["split"] == "test"][:args.n_synthetic]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(partial(real_one, src=args.src), pick, chunksize=8))
        syn = list(ex.map(partial(synth_one, root=args.synthetic), srows, chunksize=8))
    t, mech = Counter((c, g) for c, g, _ in res), defaultdict(list)
    for c, _, m in res:
        mech[c].append(m)
    classes = sorted({c for c, _, _ in res})
    print("class".ljust(14), *[g.rjust(6) for g in G], "  agree  mech%")
    ok = 0
    for c in classes:
        good = ("A", "B") if c.startswith("fresh") else ("C", "Reject")
        n = sum(t[(c, g)] for g in G)
        a = sum(t[(c, g)] for g in good)
        ok += a
        print(c.ljust(14), *[str(t[(c, g)]).rjust(6) for g in G], f"  {a:>2}/{n}  {np.mean(mech[c]):5.1f}")
    print(f"overall agreement {ok}/{len(res)}")
    i, u, e, f = map(np.array, zip(*syn))
    print(f"synthetic ({len(syn)}): defect IoU {i.sum() / u.sum():.3f}  severity MAE {e.mean():.2f}  "
          f"fruit IoU {f.mean():.3f}")


if __name__ == "__main__":
    main()
