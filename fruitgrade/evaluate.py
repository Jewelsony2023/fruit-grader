"""
Evaluate and COMPARE the multi-task model with the classification-only baseline.

    python -m fruitgrade.evaluate --multitask runs/multitask_efficientnet_b0/best.pt \
                                  --baseline  runs/baseline_efficientnet_b0/best.pt

    # only on hand-verified masks (recommended for the report):
    python -m fruitgrade.evaluate ... --mask-quality verified

Outputs (reports/ by default):
    metrics.json            every number
    comparison.md           the table for your report / slides
    confusion_<model>.png   confusion matrices
    samples.png             qualitative grid: input | GT mask | predicted mask | Grad-CAM

Headline metrics get a 95% bootstrap confidence interval: the test images are
resampled with replacement (--bootstrap times, default 1000), the metric is
recomputed each time, and the interval is the 2.5th-97.5th percentile.

How the baseline gets IoU / MAE at all:
    it has no mask, so we use the usual weakly-supervised trick — threshold its
    Grad-CAM for "Reject" (>0.5) inside the fruit as the "defect mask". That is
    exactly the kind of coarse localisation the multi-task model should beat.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fruitgrade.config import load_config
from fruitgrade.data.dataset import FruitDataset
from fruitgrade.explain import GradCAM, mc_dropout
from fruitgrade.grading import colourise_mask, grade_from_severity, severity_from_mask
from fruitgrade.inference import heatmap_overlay
from fruitgrade.metrics import ClsMeter, MAEMeter, SegMeter, bootstrap_ci
from fruitgrade.models import load_checkpoint
from fruitgrade.train import get_device

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def to_rgb(t):
    return (np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1) * 255).astype(np.uint8)


def evaluate_model(ckpt, cfg, ds, device, n_samples=6, mc_passes=10, n_boot=1000, boot_seed=0):
    model, ck = load_checkpoint(ckpt, device)
    kind, grades = ck["kind"], cfg["grades"]
    cam = GradCAM(model)
    cls, seg = ClsMeter(len(grades)), SegMeter(len(cfg["seg_classes"]))
    mae_head, mae_mask = MAEMeter(), MAEMeter()
    rule_agree, flagged, flagged_wrong, wrong = 0, 0, 0, 0
    samples = []
    reject = grades.index("Reject")
    icfg = cfg["inference"]

    for i in range(len(ds)):
        b = ds[i]
        x = b["image"][None].to(device)
        gt_mask = b["mask"].numpy()
        true_pct = float(b["severity"]) * 100
        mc = mc_dropout(model, x, mc_passes)
        pred = int(mc["top"][0])
        cls.update([int(b["grade"])], [pred])

        conf, std = float(mc["probs"][0, pred]), float(mc["top_std"][0])
        is_flag = conf < icfg["review_confidence"] or std > icfg["review_std"]
        is_wrong = pred != int(b["grade"])
        flagged += is_flag
        wrong += is_wrong
        flagged_wrong += is_flag and is_wrong

        if kind == "multitask":
            pm = mc["seg_logits"][0].argmax(0).cpu().numpy().astype(np.uint8)
            mae_head.update([true_pct], [float(mc["severity_mean"][0]) * 100])
            heat, _ = cam(x, "grade", pred)
            sev = severity_from_mask(pm)
            rule_agree += grade_from_severity(sev, cfg["grading_rules"], grades) == pred
        else:
            heat, _ = cam(x, "grade", reject)
            pm = np.where(gt_mask > 0, 1, 0).astype(np.uint8)   # fruit region from GT (generous to baseline)
            pm[(gt_mask > 0) & (heat > 0.5)] = 3
            sev = severity_from_mask(pm)
        mae_mask.update([true_pct], [sev["total_pct"]])
        seg.update(gt_mask, pm)

        if len(samples) < n_samples and i % max(1, len(ds) // n_samples) == 0:
            rgb = to_rgb(b["image"])
            samples.append({"rgb": rgb, "gt": gt_mask, "pred": pm, "cam": heat,
                            "gt_grade": grades[int(b["grade"])], "pred_grade": grades[pred],
                            "true_pct": true_pct, "pred_pct": sev["total_pct"]})

    out = {"kind": kind, "checkpoint": str(ckpt), "n": len(ds),
           "classification": cls.result(grades),
           "segmentation": seg.result(cfg["seg_classes"]),
           "severity_mae_from_mask": mae_mask.result(),
           "review_flag_rate": flagged / len(ds),
           "errors_caught_by_flag": (flagged_wrong / wrong) if wrong else None}
    if kind == "multitask":
        out["severity_mae_regression_head"] = mae_head.result()
        out["classifier_vs_mask_rule_agreement"] = rule_agree / len(ds)
    if n_boot:
        out["ci95"] = bootstrap_ci(len(ds), lambda idx: headline_metrics(kind, cfg, cls, seg, mae_mask, mae_head, idx),
                                   n_boot, boot_seed)
        out["bootstrap"] = {"resamples": n_boot, "seed": boot_seed, "unit": "image"}
    cam.remove()
    return out, samples


def headline_metrics(kind, cfg, cls, seg, mae_mask, mae_head, idx=None):
    """The metrics that get a CI, recomputed on the images idx (None = all images)."""
    c, s = cls.result(cfg["grades"], idx), seg.result(cfg["seg_classes"], idx)
    d = {"accuracy": c["accuracy"], "macro_f1": c["macro_f1"],
         "mean_defect_iou": s["mean_defect_iou"], "mean_defect_dice": s["mean_defect_dice"],
         "any_defect_iou": s["any_defect_iou"], "any_defect_dice": s["any_defect_dice"],
         "severity_mae_from_mask": mae_mask.result(idx)}
    if kind == "multitask":
        d["severity_mae_regression_head"] = mae_head.result(idx)
    return d


def plot_confusion(cm, names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_sample_grid(samples, path, label):
    rows = []
    for s in samples:
        def tag(img, text):
            img = img.copy()
            cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            return img
        rows.append(np.hstack([
            tag(s["rgb"], f"GT {s['gt_grade']} {s['true_pct']:.1f}%"),
            tag(colourise_mask(s["gt"], True), "GT mask"),
            tag(colourise_mask(s["pred"], True), f"{label}: {s['pred_grade']} {s['pred_pct']:.1f}%"),
            tag(heatmap_overlay(s["rgb"], s["cam"]), "Grad-CAM"),
        ]))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def comparison_table(results):
    def f(v, pct=False):
        if v is None:
            return "—"
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"

    def ci(r, key, v, pct=False, digits=3):
        """'value [low, high]'; the interval is added when r has a bootstrap CI for key."""
        if v is None:
            return "—"
        txt = f(v, True) if pct else f"{v:.{digits}f}"
        if key in r.get("ci95", {}):
            lo, hi = r["ci95"][key]
            txt += f" [{lo * 100:.1f}, {hi * 100:.1f}]" if pct else f" [{lo:.{digits}f}, {hi:.{digits}f}]"
        return txt

    lines = ["| Metric | " + " | ".join(r["kind"] for r in results) + " |",
             "|---|" + "---|" * len(results)]
    rows = [
        ("Grade accuracy", lambda r: ci(r, "accuracy", r["classification"]["accuracy"], pct=True)),
        ("Grade macro-F1", lambda r: ci(r, "macro_f1", r["classification"]["macro_f1"])),
        ("Mean defect IoU", lambda r: ci(r, "mean_defect_iou", r["segmentation"]["mean_defect_iou"])),
        ("Mean defect Dice", lambda r: ci(r, "mean_defect_dice", r["segmentation"]["mean_defect_dice"])),
        ("Any-defect IoU", lambda r: ci(r, "any_defect_iou", r["segmentation"]["any_defect_iou"])),
        ("Any-defect Dice", lambda r: ci(r, "any_defect_dice", r["segmentation"]["any_defect_dice"])),
        ("Severity MAE (mask, % pts)", lambda r: ci(r, "severity_mae_from_mask", r["severity_mae_from_mask"], digits=2)),
        ("Severity MAE (regression head, % pts)",
         lambda r: ci(r, "severity_mae_regression_head", r.get("severity_mae_regression_head"), digits=2)),
        ("Flagged for manual review", lambda r: f(r["review_flag_rate"], True)),
        ("Wrong grades caught by flag", lambda r: f(r["errors_caught_by_flag"], True)),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in results) + " |")
    note = ("\nValues in brackets are 95% bootstrap confidence intervals (images resampled with replacement)."
            "\nBaseline IoU/Dice/MAE use its Grad-CAM (\"Reject\" class, >0.5) inside the fruit as a proxy "
            "defect mask, and only as a single 'defect' class (counted as rot) — so per-type IoU is not "
            "meaningful for it.")
    return "\n".join(lines) + "\n" + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multitask", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-quality", default=None, help="e.g. 'verified' or 'exact,verified'")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples for 95%% CIs (0 = off)")
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    mq = args.mask_quality.split(",") if args.mask_quality else None
    ds = FruitDataset(cfg["data"]["root"], args.split, cfg["grades"], cfg["data"]["image_size"],
                      train=False, mask_quality=mq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating on {len(ds)} '{args.split}' images")

    results = []
    for ckpt in (args.multitask, args.baseline):
        if not ckpt:
            continue
        r, samples = evaluate_model(ckpt, cfg, ds, device, mc_passes=args.mc_passes,
                                    n_boot=args.bootstrap, boot_seed=args.boot_seed)
        results.append(r)
        plot_confusion(r["classification"]["confusion_matrix"], cfg["grades"],
                       f"{r['kind']} — acc {r['classification']['accuracy']:.1%}",
                       out / f"confusion_{r['kind']}.png")
        save_sample_grid(samples, out / f"samples_{r['kind']}.png", r["kind"])
        print(json.dumps({k: v for k, v in r.items() if k not in ("classification",)}, indent=1)[:1500])

    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    table = comparison_table(results)
    (out / "comparison.md").write_text(f"# Test results ({len(ds)} images, split={args.split})\n\n{table}\n",
                                       encoding="utf-8")
    print("\n" + table)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
