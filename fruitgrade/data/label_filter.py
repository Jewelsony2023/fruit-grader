"""
Label-consistency filter for TRAIN and VAL rows (test rows are never touched).

A pseudo grade that contradicts the image-level Kaggle label is most likely a
pseudo-mask failure, so the row is kept in labels.csv but flagged:
    fresh-labelled photo, pseudo grade Reject  -> "fresh_label_but_reject"
    rotten-labelled photo, pseudo grade A      -> "rotten_label_but_A"
FruitDataset (via read_labels) skips flagged rows. Nothing is deleted.

Usage (re-flag an existing labels.csv in place; prepare_kaggle already does this):
    python -m fruitgrade.data.label_filter --root data/processed [--audit data/interim/dataset_audit.json]
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from fruitgrade.data.synthetic import write_labels

FILTER_SPLITS = ("train", "val")


def exclusion_reason(row) -> str:
    if row["split"] not in FILTER_SPLITS:
        return ""
    if row["source"] == "kaggle_fresh" and row["grade"] == "Reject":
        return "fresh_label_but_reject"
    if row["source"] == "kaggle_rotten" and row["grade"] == "A":
        return "rotten_label_but_A"
    return ""


def apply_filter(rows):
    for r in rows:
        r["excluded_reason"] = exclusion_reason(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--audit", default=None, help="also record the counts in this audit json")
    args = ap.parse_args()
    path = Path(args.root) / "labels.csv"
    rows = apply_filter(list(csv.DictReader(open(path, encoding="utf-8"))))
    write_labels(path, rows)
    ex = [r for r in rows if r["excluded_reason"]]
    print("excluded:", dict(Counter((r["split"], r["excluded_reason"]) for r in ex)))
    if args.audit:
        cls = lambda r: r["source"].replace("kaggle_", "") + "_" + r["fruit"]  # noqa: E731
        p = Path(args.audit)
        audit = json.loads(p.read_text())
        audit["label_consistency_filter"] = {
            "rule": "train/val only: fresh label + pseudo Reject, or rotten label + pseudo A -> excluded",
            "excluded": len(ex),
            "by_reason_split": {f"{k[0]}/{k[1]}": v for k, v in
                                sorted(Counter((r["excluded_reason"], r["split"]) for r in ex).items())},
            "by_class_split": {f"{k[0]}/{k[1]}": v for k, v in
                               sorted(Counter((cls(r), r["split"]) for r in ex).items())},
            "remaining_split_grade": {s: dict(sorted(Counter(r["grade"] for r in rows
                                                             if r["split"] == s and not r["excluded_reason"]).items()))
                                      for s in ("train", "val", "test")}}
        p.write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
