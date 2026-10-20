"""
Synthetic fruit generator — EXACT pixel masks for free.

Why this exists
---------------
Public fresh/rotten datasets have no pixel masks and no A/B/C grades. This
generator draws fruit-like images with bruises, rot and mechanical damage and
writes the exact mask, so you can:
  * smoke-test the whole pipeline in minutes (no downloads),
  * sanity-check that the segmentation head CAN learn the defect types,
  * demo the web app.
DO NOT report synthetic numbers as your project results — use the Kaggle
pseudo-labelled data + a hand-verified test subset for that.

Usage:
    python -m fruitgrade.data.synthetic --n 1200 --out data/processed
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from fruitgrade.config import load_config
from fruitgrade.grading import (BRUISE, HEALTHY, MECHANICAL, ROT,
                                grade_from_severity, severity_from_mask)

FRUITS = {
    #  name     base RGB colours to pick from               shape (rx, ry) ratio
    "apple":  ([(190, 30, 35), (170, 40, 30), (120, 180, 50), (210, 170, 40)], (1.00, 0.92)),
    "orange": ([(240, 140, 20), (230, 120, 10), (245, 160, 40)],               (1.00, 1.00)),
    "pear":   ([(170, 190, 60), (190, 180, 70), (150, 170, 50)],               (0.80, 1.00)),
    "tomato": ([(215, 40, 30), (200, 50, 25)],                                 (1.00, 0.88)),
}


def _smooth_noise(h, w, scale, rng):
    """Low-frequency noise in [0,1] (used for texture and irregular blobs)."""
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), scale / 2)
    big -= big.min()
    return big / (big.max() + 1e-6)


def _background(size, rng):
    kind = rng.integers(3)
    if kind == 0:   # plain light (like a packhouse tray)
        c = rng.integers(200, 250, 3)
        bg = np.ones((size, size, 3), np.float32) * c
    elif kind == 1:  # dark conveyor
        c = rng.integers(20, 70, 3)
        bg = np.ones((size, size, 3), np.float32) * c
    else:           # textured wood/cloth-ish
        base = rng.integers(90, 200, 3).astype(np.float32)
        n = _smooth_noise(size, size, 8, rng)[..., None]
        bg = base * (0.75 + 0.5 * n)
    return bg


def _irregular_blob(size, cx, cy, r, rng, roughness=0.35):
    """Binary-ish soft blob with an irregular edge. Returns float alpha 0..1."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1)
    n = _smooth_noise(size, size, max(4, int(r / 2)), rng)
    d = d + roughness * (n - 0.5) * 2
    return np.clip(1.0 - d, 0, 1)


def make_sample(size=256, target_grade=None, rng=None, rules=None):
    rng = rng or np.random.default_rng()
    fruit_name = rng.choice(list(FRUITS))
    colours, (ax, ay) = FRUITS[fruit_name]
    base = np.array(colours[rng.integers(len(colours))], np.float32)

    img = _background(size, rng)
    mask = np.zeros((size, size), np.uint8)

    # ---- fruit body -------------------------------------------------------
    cx = size / 2 + rng.uniform(-0.08, 0.08) * size
    cy = size / 2 + rng.uniform(-0.08, 0.08) * size
    R = rng.uniform(0.30, 0.40) * size
    rx, ry = R * ax, R * ay
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    edge_noise = 0.04 * (_smooth_noise(size, size, 16, rng) - 0.5)
    dist = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2) + edge_noise
    fruit = dist < 1.0
    mask[fruit] = HEALTHY

    # colour + texture + 3D shading + specular highlight
    tex = _smooth_noise(size, size, 6, rng)[..., None]
    shade = np.clip(1.15 - 0.55 * dist, 0.45, 1.15)[..., None]
    lx, ly = cx - 0.35 * rx, cy - 0.35 * ry
    spec = np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * (0.18 * R) ** 2)))[..., None]
    fruit_rgb = base * (0.85 + 0.3 * tex) * shade + 90 * spec
    img = np.where(fruit[..., None], fruit_rgb, img)

    # soft shadow under the fruit
    shadow = np.clip(1 - np.sqrt(((xx - cx - 6) / (rx * 1.1)) ** 2 + ((yy - cy - 8) / (ry * 1.1)) ** 2), 0, 1)
    img = np.where(fruit[..., None], img, img * (1 - 0.35 * shadow[..., None]))

    # ---- choose how much damage to draw ----------------------------------
    target_grade = rng.integers(4) if target_grade is None else target_grade
    target_total = {0: rng.uniform(0, 1.2), 1: rng.uniform(3, 9),
                    2: rng.uniform(11, 26), 3: rng.uniform(32, 55)}[target_grade]
    fruit_px = fruit.sum()

    def inside_point(margin=0.75):
        while True:
            px, py = rng.uniform(cx - rx, cx + rx), rng.uniform(cy - ry, cy + ry)
            if dist[int(np.clip(py, 0, size - 1)), int(np.clip(px, 0, size - 1))] < margin:
                return px, py

    def current_total():
        return 100.0 * ((mask >= BRUISE).sum()) / fruit_px

    # defect mix depends on target grade (Reject fruit usually has rot)
    if target_grade == 0:
        kinds = ["none"]
    elif target_grade == 1:
        kinds = ["bruise", "mechanical"]
    elif target_grade == 2:
        kinds = ["bruise", "mechanical", "bruise", "rot_small"]
    else:
        kinds = ["rot", "bruise", "rot", "mechanical"]

    guard = 0
    while current_total() < target_total and guard < 12 and kinds != ["none"]:
        guard += 1
        kind = kinds[rng.integers(len(kinds))]
        remaining = (target_total - current_total()) / 100 * fruit_px
        r = np.sqrt(max(remaining, 30) / np.pi) * rng.uniform(0.6, 1.0)
        r = float(np.clip(r, 4, 0.7 * R))
        px, py = inside_point()

        if kind == "bruise":
            a = _irregular_blob(size, px, py, r, rng, 0.25)
            a = cv2.GaussianBlur(a, (0, 0), 2.0) * fruit
            # bruise = darker, brownish-purple, soft edge (visible wherever mask says bruise)
            darker = img * np.array([0.42, 0.36, 0.40]) + np.array([45, 25, 35])
            k = np.clip(a * 1.8, 0, 0.95)[..., None]
            img = img * (1 - k) + darker * k
            sel = (a > 0.35) & fruit & (mask != ROT)
            mask[sel] = BRUISE

        elif kind in ("rot", "rot_small"):
            if kind == "rot_small":
                r = min(r, 0.12 * R)
            a = _irregular_blob(size, px, py, r, rng, 0.45) * fruit
            rings = 0.5 + 0.5 * np.sin(np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / 2.2)
            rot_rgb = np.stack([70 + 30 * rings, 40 + 15 * rings, 15 + 8 * rings], -1)
            # white/green mould specks
            specks = (rng.random((size, size)) > 0.985) & (a > 0.4)
            specks = cv2.dilate(specks.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
            rot_rgb[specks] = (200, 210, 180) if rng.random() < 0.6 else (110, 150, 80)
            hard = np.clip((a - 0.1) * 5, 0, 1)[..., None]
            img = img * (1 - hard) + rot_rgb * hard
            mask[(a > 0.2) & fruit] = ROT

        elif kind == "mechanical":
            # cuts / scratches: polylines with pale flesh centre and dark rim
            layer = np.zeros((size, size), np.uint8)
            n_cuts = rng.integers(1, 4)
            for _ in range(n_cuts):
                x0, y0 = inside_point(0.8)
                L = rng.uniform(0.3, 0.9) * R
                th = rng.uniform(0, np.pi)
                pts = []
                for t in np.linspace(0, 1, 6):
                    pts.append([x0 + t * L * np.cos(th) + rng.normal(0, 2),
                                y0 + t * L * np.sin(th) + rng.normal(0, 2)])
                w = int(rng.integers(2, 6))
                cv2.polylines(layer, [np.int32(pts)], False, 255, w)
            layer = (layer > 0) & fruit
            rim = cv2.dilate(layer.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & fruit
            img[rim] = img[rim] * 0.45
            flesh = np.array([235, 220, 170], np.float32)
            img[layer] = flesh * rng.uniform(0.8, 1.0)
            mask[rim & (mask != ROT)] = MECHANICAL

    # camera noise + slight blur
    img = img + rng.normal(0, 4, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0.6)

    sev = severity_from_mask(mask)
    grade = grade_from_severity(sev, rules) if rules else target_grade
    return img, mask, sev, grade, fruit_name


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic fruit dataset")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default=None, help="default: data.root from config")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["data"]["root"])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    grades = cfg["grades"]

    rows = []
    for i in range(args.n):
        img, mask, sev, g, fruit = make_sample(args.size, i % 4, rng, cfg["grading_rules"])
        name = f"syn_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)
        r = rng.random()
        split = "train" if r < 0.7 else ("val" if r < 0.85 else "test")
        rows.append({"image": f"images/{name}.jpg", "mask": f"masks/{name}.png",
                     "grade": grades[g], "severity": round(sev["total_pct"], 3),
                     "split": split, "fruit": fruit, "source": "synthetic",
                     "mask_quality": "exact"})

    write_labels(out / "labels.csv", rows)
    from collections import Counter
    print(f"Wrote {len(rows)} samples to {out}")
    print("grades:", Counter(r["grade"] for r in rows))
    print("splits:", Counter(r["split"] for r in rows))


LABEL_FIELDS = ["image", "mask", "grade", "severity", "split", "fruit", "source", "mask_quality", "group", "src",
                "excluded_reason"]


def write_labels(path, rows, append=False):
    path = Path(path)
    exists = path.exists() and append
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
