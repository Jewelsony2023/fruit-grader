"""
Dataset audit + leak-free train/val/test split (Phase 1).

The Kaggle fresh/rotten set is 1,511 original photos, each saved 9 times:
the original, rotated by 15/30/45/60/75 degrees, salt-and-pepper noise,
a translation and a vertical flip. Kaggle's own train/test split puts
versions of the same photo on both sides, so it cannot be used.

What this script does:
1. Keeps ONLY the originals (all Kaggle augmented copies are dropped: noise and
   black rotation corners corrupt pseudo-masks, and dataset.py augments anyway).
   Every original is one group.
2. Hashes the originals with a pHash that ignores rotations/flips (minimum over
   the 8 rotated/flipped versions) and lists CROSS-ORIGINAL pairs within
   --max-dist bits, with a contact sheet to check them by eye.
3. Merges only the pairs marked "yes" in a decisions CSV (--merges), never on the
   threshold alone (project rule: when unsure, merge). A pair marked "conflict"
   (same photo/fruit, contradicting labels) drops BOTH images.
4. Splits groups train/val/test, stratified by class (fresh/rotten x fruit).

Usage:
    # round 1: audit, candidate pairs, contact sheet (no merges yet)
    python -m fruitgrade.data.dedupe --src D:/datasets/fruits/dataset --out data/interim
    # round 2: after filling in data/interim/merge_decisions.csv (pair_id,merge,note)
    python -m fruitgrade.data.dedupe --src ... --out data/interim --merges data/interim/merge_decisions.csv

Writes to <out>: splits.csv (one row per original), candidate_pairs.csv,
contact_sheet_<k>.png, dataset_audit.json.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from fruitgrade.data.prepare_kaggle import IMG_EXT, parse_path

AUG_RE = re.compile(r"^(rotated_by_-?\d+_|saltandpepper_|translation_|vertical_flip_)")
DIHEDRAL = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
            Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
            Image.Transpose.TRANSVERSE]


def invariant_phash(img: Image.Image) -> int:
    """Minimum 64-bit pHash over the 8 rotations/flips of the image."""
    # pHash works on a 32x32 greyscale image; shrinking once first is much faster
    # and resizing to a square commutes with rotations/flips.
    small = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    return min(int(str(imagehash.phash(small if t is None else small.transpose(t))), 16)
               for t in DIHEDRAL)


def near_pairs(hashes, max_dist):
    """All pairs (i, j, dist) with i < j and Hamming distance <= max_dist. No chaining."""
    h = np.array(hashes, dtype=np.uint64)
    d = np.bitwise_count(h[:, None] ^ h[None, :])
    i, j = np.nonzero(np.triu(d <= max_dist, k=1))
    return [(int(a), int(b), int(d[a, b])) for a, b in zip(i, j)]


def original_key(rel: Path):
    """'train/freshapples/rotated_by_15_X.png' -> ('freshapples/X.png', 'rotated_by_15_')."""
    m = AUG_RE.match(rel.name)
    aug = m.group(1) if m else ""
    return f"{rel.parent.name}/{rel.name[len(aug):]}", aug


def split_groups(group_cls, group_size, fracs, seed):
    """Group -> split, stratified by class; whole groups move together."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for g, c in group_cls.items():
        by_class[c].append(g)
    out = {}
    for cls in sorted(by_class):
        gids = sorted(by_class[cls])
        rng.shuffle(gids)
        total = sum(group_size[g] for g in gids)
        k = 0
        for name in ("test", "val"):
            filled, target = 0, round(fracs[name] * total)
            while k < len(gids) and filled < target:
                out[gids[k]] = name
                filled += group_size[gids[k]]
                k += 1
        for g in gids[k:]:
            out[g] = "train"
    return out


def contact_sheets(src, rows, pairs, out, per_page=24, thumb=150):
    """Each line: pair id + distance, image A | image B (with class). Returns file names."""
    def tile(r):
        img = cv2.imread(str(src / r["path"]))
        img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        bar = np.full((18, thumb, 3), 255, np.uint8)
        cv2.putText(bar, r["cls"], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    files, cols = [], 3                                   # 3 pairs per line
    for p0 in range(0, len(pairs), per_page):
        cells = []
        for pid, a, b, d in pairs[p0:p0 + per_page]:
            head = np.full((18, 2 * thumb + 4, 3), 230, np.uint8)
            cv2.putText(head, f"#{pid}  dist {d}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 160), 1, cv2.LINE_AA)
            sep = np.full((thumb + 18, 4, 3), 255, np.uint8)
            cell = np.vstack([head, np.hstack([tile(rows[a]), sep, tile(rows[b])])])
            cells.append(np.pad(cell, ((6, 6), (6, 6), (0, 0)), constant_values=255))
        while len(cells) % cols:
            cells.append(np.full_like(cells[0], 255))
        sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        name = f"contact_sheet_{p0 // per_page + 1}.png"
        cv2.imwrite(str(out / name), sheet)
        files.append(name)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ONE dataset tree (with train/ and test/)")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--merges", default=None, help="CSV pair_id,merge(yes/no),note from a visual check")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. versions per original, and where Kaggle put them
    versions = defaultdict(list)                          # key -> [(aug, kaggle_split, rel)]
    for p in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
        rel = p.relative_to(src)
        info = parse_path(rel)
        if info is None:
            continue
        key, aug = original_key(rel)
        versions[key].append((aug, "test" if info["is_test"] else "train", rel.as_posix()))

    rows, no_original = [], []
    for key in sorted(versions):
        plain = [v for v in versions[key] if v[0] == ""]
        if len(plain) != 1:
            no_original.append(key)
            continue
        info = parse_path(Path(plain[0][2]))
        rows.append({"path": plain[0][2], "key": key,
                     "cls": ("rotten" if info["rotten"] else "fresh") + "_" + info["fruit"],
                     "kaggle_split": plain[0][1]})
    for r in rows:
        with Image.open(src / r["path"]) as im:
            r["phash"] = f"{invariant_phash(im):016x}"

    # 2. cross-original candidate pairs (listed, not chained)
    pairs = [(pid, a, b, d) for pid, (a, b, d) in
             enumerate(near_pairs([int(r["phash"], 16) for r in rows], args.max_dist), 1)]
    with open(out / "candidate_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "dist", "path_a", "cls_a", "path_b", "cls_b"])
        for pid, a, b, d in pairs:
            w.writerow([pid, d, rows[a]["path"], rows[a]["cls"], rows[b]["path"], rows[b]["cls"]])
    sheets = contact_sheets(src, rows, pairs, out)

    # 3. merge only visually confirmed pairs
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged, conflicts, drop = [], [], set()
    if args.merges:
        decision = {int(d["pair_id"]): d["merge"].strip().lower()
                    for d in csv.DictReader(open(args.merges, encoding="utf-8"))}
        for pid, a, b, d in pairs:
            entry = {"pair_id": pid, "dist": d, "a": rows[a]["path"], "b": rows[b]["path"]}
            if decision.get(pid) == "yes":
                parent[max(find(a), find(b))] = min(find(a), find(b))
                merged.append(entry)
            elif decision.get(pid) == "conflict":
                conflicts.append({**entry, "cls_a": rows[a]["cls"], "cls_b": rows[b]["cls"]})
                drop |= {a, b}
    for i, r in enumerate(rows):
        r["group"] = find(i)
    n_originals = len(rows)
    rows = [r for i, r in enumerate(rows) if i not in drop]

    # 4. stratified group split
    members = defaultdict(list)
    for r in rows:
        members[r["group"]].append(r)
    group_cls = {g: Counter(r["cls"] for r in m).most_common(1)[0][0] for g, m in members.items()}
    fracs = {"val": args.val_frac, "test": args.test_frac}
    gsplit = split_groups(group_cls, {g: len(m) for g, m in members.items()}, fracs, args.seed)
    for r in rows:
        r["split"] = gsplit[r["group"]]
    assert all(len({r["split"] for r in m}) == 1 for m in members.values())

    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "cls", "kaggle_split", "phash", "group", "split"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)

    n_versions = Counter(len(v) for v in versions.values())
    leaking = [k for k, v in versions.items() if len({s for _, s, _ in v}) > 1]
    audit = {
        "src": str(src), "seed": args.seed, "max_dist": args.max_dist,
        "files_total": sum(len(v) for v in versions.values()),
        "originals": n_originals,
        "keys_without_exactly_one_original": len(no_original),
        "versions_per_original_hist": dict(sorted(n_versions.items())),
        "augmentation_type_counts": dict(sorted(Counter(
            re.sub(r"\d+", "N", a) or "original" for v in versions.values() for a, _, _ in v).items())),
        "kaggle_split_leak_originals": len(leaking),
        "kaggle_originals_by_split": dict(Counter(r["kaggle_split"] for r in rows)),
        "candidate_pairs_within_max_dist": len(pairs),
        "originals_in_candidate_pairs": len({i for _, a, b, _ in pairs for i in (a, b)}),
        "candidate_pairs_cross_class": sum(rows[a]["cls"] != rows[b]["cls"] for _, a, b, _ in pairs),
        "merge_decisions_file": args.merges,
        "merged_pairs": merged,
        "label_conflict": conflicts,
        "label_conflict_images_dropped": len(drop),
        "originals_used": len(rows),
        "groups": len(members),
        "biggest_group": max(len(m) for m in members.values()),
        "mixed_class_groups": sum(len({r["cls"] for r in m}) > 1 for m in members.values()),
        "split_images": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "split_groups": {s: sum(v == s for v in gsplit.values()) for s in ("train", "val", "test")},
        "split_by_class": {s: dict(sorted(Counter(r["cls"] for r in rows if r["split"] == s).items()))
                           for s in ("train", "val", "test")},
        "contact_sheets": sheets,
    }
    audit_path = out / "dataset_audit.json"
    if audit_path.exists():                               # keep entries other tools added (e.g. pm_eval)
        audit = {**json.loads(audit_path.read_text()), **audit}
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "merged_pairs"}, indent=2))
    print("merged pairs:", len(merged), "| label conflicts:", len(conflicts),
          "| images dropped:", len(drop))


if __name__ == "__main__":
    main()
