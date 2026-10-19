"""
Severity + grade rules.

Severity (%) = defect pixels / fruit pixels * 100
    fruit pixels  = healthy + bruise + rot + mechanical
    defect pixels = bruise + rot + mechanical

The grade rule is deliberately simple and written in config.yaml so it can be
defended in a viva and changed without touching code.
"""
import numpy as np

BACKGROUND, HEALTHY, BRUISE, ROT, MECHANICAL = 0, 1, 2, 3, 4
DEFECT_IDS = {"bruise": BRUISE, "rot": ROT, "mechanical": MECHANICAL}

# RGB colours used everywhere a mask is drawn (UI + saved figures)
MASK_COLOURS = {
    BACKGROUND: (0, 0, 0),
    HEALTHY: (60, 170, 90),       # green
    BRUISE: (155, 89, 182),       # purple
    ROT: (200, 40, 40),           # red
    MECHANICAL: (240, 180, 20),   # amber
}


def severity_from_mask(mask: np.ndarray) -> dict:
    """Return total and per-defect severity (% of fruit surface) for a class-id mask."""
    fruit = int((mask != BACKGROUND).sum())
    out = {"fruit_pixels": fruit}
    if fruit == 0:
        out.update(total_pct=0.0, bruise_pct=0.0, rot_pct=0.0, mechanical_pct=0.0)
        return out
    total = 0
    for name, cid in DEFECT_IDS.items():
        n = int((mask == cid).sum())
        total += n
        out[f"{name}_pct"] = 100.0 * n / fruit
    out["total_pct"] = 100.0 * total / fruit
    return out


def grade_from_severity(sev: dict, rules: dict, grades=("A", "B", "C", "Reject")) -> int:
    """Apply the config rules. Returns the grade index (0=A ... 3=Reject)."""
    total, rot = sev["total_pct"], sev.get("rot_pct", 0.0)
    r = rules["reject"]
    if total >= r["total_pct"] or rot >= r["rot_pct"]:
        return grades.index("Reject")
    c = rules["c"]
    if total >= c["total_pct"] or rot >= c["rot_pct"]:
        return grades.index("C")
    if total >= rules["b"]["total_pct"]:
        return grades.index("B")
    return grades.index("A")


def colourise_mask(mask: np.ndarray, show_healthy: bool = False) -> np.ndarray:
    """Class-id mask -> RGB image."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, col in MASK_COLOURS.items():
        if cid == HEALTHY and not show_healthy:
            continue
        rgb[mask == cid] = col
    return rgb
