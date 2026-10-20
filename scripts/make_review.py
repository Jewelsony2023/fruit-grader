"""
Build results/review/ for the project review. Every number is read from a file
(named next to it), nothing is typed in by hand.

    python scripts/make_review.py [--prelim results/preliminary] [--out results/review]

Contents:
    README.md                    index + all tables (data audit, class x grade, preliminary comparison)
    pseudo_mask_before_after.jpg one contact sheet (fresh apples) before vs after tuning
    preliminary/                 comparison table, confusion matrices, sample grids (if the run finished)
    app/                         app screenshots (written by scripts/app_screenshots.py)
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRELIM = "PRELIMINARY — test ground truth is auto-generated pseudo-masks, not human-verified"
G = ["A", "B", "C", "Reject"]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def before_after(out, cls="fresh_apple"):
    a = cv2.imread(str(ROOT / f"data/interim/sheets_before/{cls}.jpg"))
    b = cv2.imread(str(ROOT / f"data/interim/sheets_after/{cls}.jpg"))
    if a is None or b is None:
        return None

    def bar(text, w):
        im = np.full((50, w, 3), 255, np.uint8)
        cv2.putText(im, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        return im
    w = max(a.shape[1], b.shape[1])
    sheet = np.vstack([bar("BEFORE tuning (same 24 images)", w), a, bar("AFTER tuning", w), b])
    cv2.imwrite(str(out / "pseudo_mask_before_after.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "pseudo_mask_before_after.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim", default="results/preliminary")
    ap.add_argument("--out", default="results/review")
    args = ap.parse_args()
    out, prelim = ROOT / args.out, ROOT / args.prelim
    out.mkdir(parents=True, exist_ok=True)

    audit_f = "results/data_audit/dataset_audit.json"
    a = json.loads((ROOT / audit_f).read_text())
    vers = a["versions_per_original_hist"]
    vers_txt = (f"{a['originals']:,} original photos x {next(iter(vers))} versions each" if len(vers) == 1
                else ", ".join(f"{v} originals with {k} versions" for k, v in vers.items()))
    md = ["# Review pack: fruit-grader", "",
          f"Every number below is read from the file named next to it. "
          f"Model results are **{PRELIM}**.", ""]

    # 1. data audit
    md += ["## 1. Data audit", f"Source: `{audit_f}`", "",
           f"- Kaggle files: **{a['files_total']:,}** = **{vers_txt}** "
           f"(original + rotations 15–75 deg, salt-and-pepper, shift, vertical flip).",
           f"- Augmentation counts: {', '.join(f'{k} {v}' for k, v in a['augmentation_type_counts'].items())}.",
           f"- Kaggle's own train/test split **leaks**: versions of **{a['kaggle_split_leak_originals']:,}** of "
           f"{a['originals']:,} originals sit on both sides. We do not use it.",
           f"- We use the **originals only** ({a['originals_used']:,}); all augmented copies are dropped.",
           f"- Near-duplicates: {a['candidate_pairs_within_max_dist']} candidate pairs (pHash <= {a['max_dist']}), "
           f"checked by eye; {len(a['merged_pairs'])} merged, {len(a['label_conflict'])} label conflicts.",
           f"- Group split (stratified by class, seed {a['seed']}): {a['groups']:,} groups; no group in two splits.",
           ""]
    sbc = a["split_by_class"]
    classes = sorted(sbc["train"])
    md += [table(["Class", "Train", "Val", "Test"],
                 [[c] + [sbc[s].get(c, 0) for s in ("train", "val", "test")] for c in classes]
                 + [["**Images**"] + [a["split_images"][s] for s in ("train", "val", "test")],
                    ["**Groups**"] + [a["split_groups"][s] for s in ("train", "val", "test")]]), ""]
    if "pseudo_mask_tuning_sample" in a:
        t = a["pseudo_mask_tuning_sample"]
        md += [f"- Caveat: the pseudo-mask tuning sample ({t['images']} images) was drawn from all splits "
               f"({', '.join(f'{k} {v}' for k, v in t['by_split'].items())}).", ""]

    # 2. pseudo-masks
    md += ["## 2. Pseudo-masks (weak labels)", ""]
    ba = before_after(out)
    if ba:
        md += [f"![before/after]({ba})", "",
               "One class shown (fresh apples): `data/interim/sheets_before|after/fresh_apple.jpg`.", ""]
    labels_f = "data/processed/labels.csv"
    rows = list(csv.DictReader(open(ROOT / labels_f, encoding="utf-8")))
    cls = lambda r: r["source"].replace("kaggle_", "") + "_" + r["fruit"]  # noqa: E731
    t = Counter((cls(r), r["grade"]) for r in rows)
    md += [f"**Class x pseudo grade**, all {len(rows):,} originals (source: `{labels_f}`)", "",
           table(["Class"] + G + ["Total"],
                 [[c] + [t[(c, g)] for g in G] + [sum(t[(c, g)] for g in G)] for c in sorted({cls(r) for r in rows})]
                 + [["**All**"] + [sum(v for (c, g2), v in t.items() if g2 == g) for g in G] + [len(rows)]]), ""]
    if "label_consistency_filter" in a:
        f = a["label_consistency_filter"]
        rem = f["remaining_split_grade"]
        md += [f"Label-consistency filter (train/val only): **{f['excluded']}** rows excluded "
               f"({', '.join(f'{k} {v}' for k, v in f['by_reason_split'].items())}). Used for training:", "",
               table(["Split"] + G + ["Total"],
                     [[s] + [rem[s].get(g, 0) for g in G] + [sum(rem[s].values())] for s in ("train", "val", "test")]),
               ""]

    # 3. preliminary results
    md += [f"## 3. {PRELIM}", ""]
    if (prelim / "comparison.md").exists():
        (out / "preliminary").mkdir(exist_ok=True)
        for p in sorted(prelim.glob("*")):
            if p.suffix in (".md", ".json", ".png", ".txt"):
                shutil.copyfile(p, out / "preliminary" / p.name)
        comp = (prelim / "comparison.md").read_text(encoding="utf-8")
        md += [f"Source: `{args.prelim}/comparison.md` (copied to `preliminary/`)", "",
               "\n".join(l for l in comp.splitlines() if not l.startswith("# ")), ""]
        if (prelim / "run_settings.json").exists():
            md += ["Run settings (`run_settings.json`):", "", "```json",
                   (prelim / "run_settings.json").read_text(encoding="utf-8").strip(), "```", ""]
        for p in sorted((out / "preliminary").glob("*.png")):
            md += [f"![{p.stem}](preliminary/{p.name})", ""]
    else:
        md += ["Not available: the preliminary run did not produce `comparison.md` (see logs/).", ""]

    # 4. app
    shots = sorted((out / "app").glob("*.png")) if (out / "app").exists() else []
    md += ["## 4. Web app (preliminary checkpoints)", ""]
    md += [f"![{p.stem}](app/{p.name})" for p in shots] or ["No screenshots (see logs/)."]
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"review pack -> {out} ({len(shots)} screenshots)")


if __name__ == "__main__":
    main()
