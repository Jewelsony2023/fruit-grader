"""
Pseudo-mask generation for images that only have a fresh/rotten label.

Pipeline (one image):
  1. Fruit vs background  -> GrabCut, seeded by Otsu on saturation + border colour.
  2. Defect candidates    -> LAB colour space: pixels much darker than the fruit's
                             own healthy colour, plus whitish/grey mould pixels.
  3. Clean up             -> morphology, drop tiny blobs, ignore thin rim (shading).
  4. Defect type per blob -> simple shape/colour rules:
                               rot        : very dark, or contains mould, or big & textured
                               mechanical : long & thin (cut / scratch)
                               bruise     : everything else (moderate, smooth darkening)
  5. (optional) Grad-CAM  -> drop blobs the trained classifier does not look at.

These masks are APPROXIMATE (weak supervision). Always hand-check a test subset
(see README: "verified test set").
"""
import cv2
import numpy as np

from fruitgrade.grading import BACKGROUND, BRUISE, HEALTHY, MECHANICAL, ROT


# ---------------------------------------------------------------- step 1
def segment_fruit(rgb: np.ndarray, iters: int = 5) -> np.ndarray:
    """Return bool mask of the (largest) fruit in the image."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # cue A: fruit is usually more saturated than the background
    sat = hsv[..., 1]
    _, sat_fg = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cue B: fruit colour differs from the image border colour
    border = np.concatenate([lab[:5].reshape(-1, 3), lab[-5:].reshape(-1, 3),
                             lab[:, :5].reshape(-1, 3), lab[:, -5:].reshape(-1, 3)])
    bg_col = np.median(border, 0)
    diff = np.linalg.norm(lab - bg_col, axis=-1)
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, col_fg = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    seed = ((sat_fg > 0) | (col_fg > 0)).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[seed > 0] = cv2.GC_PR_FGD
    sure_fg = cv2.erode(seed, np.ones((15, 15), np.uint8))
    gc[sure_fg > 0] = cv2.GC_FGD
    m = max(2, int(0.02 * min(h, w)))
    gc[:m, :] = gc[-m:, :] = cv2.GC_BGD
    gc[:, :m] = gc[:, -m:] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).sum() + (gc == cv2.GC_PR_FGD).sum() < 50:
        return np.zeros((h, w), bool)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass
    fruit = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    return _largest_component_filled(fruit)


def _largest_component_filled(binary: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    # fill holes (dark rot inside the fruit must stay part of the fruit)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return filled.astype(bool)


# ---------------------------------------------------------------- step 2-4
def detect_defects(rgb, fruit, is_rotten=None, dark_ratio=0.72, min_blob_frac=0.002):
    """Return class-id mask (0 bg, 1 healthy, 2 bruise, 3 rot, 4 mechanical)."""
    mask = np.zeros(fruit.shape, np.uint8)
    mask[fruit] = HEALTHY
    if fruit.sum() < 100:
        return mask

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.sqrt(A ** 2 + B ** 2)

    # ignore the outer rim: it is dark because of 3-D shading, not damage
    fruit_u8 = fruit.astype(np.uint8)
    rim = max(5, int(0.06 * np.sqrt(fruit.sum())))
    inner = cv2.erode(fruit_u8, np.ones((rim, rim), np.uint8)).astype(bool)

    # --- illumination correction -------------------------------------
    # A fruit is brighter in the middle and darker at the edge (3-D shading).
    # Grey "closing" with a big kernel removes dark spots smaller than the kernel
    # and keeps the smooth shading -> expected healthy brightness per pixel.
    diam = 2 * np.sqrt(fruit.sum() / np.pi)
    Lfill = L.copy()
    Lfill[~fruit] = np.median(L[fruit])
    # clip bright outliers (specular highlights, pale cuts) so they don't
    # inflate the expected brightness of their neighbourhood
    Lclip = np.minimum(Lfill, np.percentile(L[inner], 80))
    k_big = max(15, int(0.45 * diam)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_big, k_big))
    shading = cv2.morphologyEx(Lclip, cv2.MORPH_CLOSE, se)
    shading = cv2.GaussianBlur(shading, (0, 0), k_big / 4)
    Lref = np.percentile(L[inner], 70)
    dark_score = shading - L

    # thin pale lines (cuts / scratches expose pale flesh): white top-hat
    k_small = max(5, int(0.06 * diam)) | 1
    se_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    bright_thin = cv2.morphologyEx(Lfill, cv2.MORPH_TOPHAT, se_s) > 28

    Cref = np.percentile(chroma[inner], 60)
    # a pixel is "dark" if it is clearly below the brightness expected there
    # (ratio, so it works for both light and dark fruit); stricter for "fresh" images
    ratio = dark_ratio * (0.9 if is_rotten is False else 1.0)
    dark = L < ratio * shading
    # mould: whitish/grey, low chroma, blotchy (not a smooth specular highlight)
    mould = (chroma < 0.45 * Cref) & (L > Lref - 10) & (Cref > 15) & ~bright_thin
    mould = cv2.morphologyEx(mould.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    cand = (dark | mould | bright_thin) & inner
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # local texture (std of L) — rot is blotchy, bruises are smooth
    Lf = L / 255.0
    mu = cv2.blur(Lf, (7, 7))
    tex = np.sqrt(np.clip(cv2.blur(Lf ** 2, (7, 7)) - mu ** 2, 0, None)) * 255

    fruit_area = fruit.sum()
    n, cc, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_blob_frac * fruit_area:
            continue
        blob = cc == k
        dL = float(dark_score[blob].mean())
        thin_frac = float(bright_thin[blob].mean())
        mould_frac = float(mould[blob].mean())
        texture = float(tex[blob].mean())

        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        elong = max(rw, rh) / max(1.0, min(rw, rh))
        hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        solidity = cv2.contourArea(c) / hull
        thickness = area / max(1.0, max(rw, rh))

        if thin_frac > 0.2 or elong > 3.5 or (solidity < 0.45 and thickness < 6):
            cls = MECHANICAL
        elif dL > 45 or mould_frac > 0.25 or (area > 0.05 * fruit_area and texture > 9):
            cls = ROT
        elif is_rotten and dL > 32:
            cls = ROT
        else:
            cls = BRUISE
        mask[blob] = cls
    return mask


# ---------------------------------------------------------------- step 5
def refine_with_cam(mask, cam, keep_thr=0.25):
    """Drop defect blobs whose mean Grad-CAM activation is below keep_thr."""
    out = mask.copy()
    defect = (mask >= BRUISE).astype(np.uint8)
    n, cc = cv2.connectedComponents(defect, 8)
    for k in range(1, n):
        blob = cc == k
        if cam[blob].mean() < keep_thr:
            out[blob] = HEALTHY
    return out


def pseudo_mask(rgb: np.ndarray, is_rotten=None, cam=None) -> np.ndarray:
    fruit = segment_fruit(rgb)
    m = detect_defects(rgb, fruit, is_rotten)
    if cam is not None:
        m = refine_with_cam(m, cam)
    m[~fruit] = BACKGROUND
    return m
