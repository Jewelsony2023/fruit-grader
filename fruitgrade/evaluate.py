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


def predict_image(model, kind, cam, b, cfg, mc_passes, device):
    """Everything evaluate_model needs for ONE dataset item b. Shared with
    fruitgrade.predict_split so the per-image CSV and metrics.json come from the same code."""
    grades, icfg = cfg["grades"], cfg["inference"]
    x = b["image"][None].to(device)
    gt_mask = b["mask"].numpy()
    mc = mc_dropout(model, x, mc_passes)
    pred = int(mc["top"][0])
    conf, std = float(mc["probs"][0, pred]), float(mc["top_std"][0])
    r = {"pred": pred, "probs": mc["probs"][0].cpu().numpy(), "conf": conf, "std": std,
         "entropy": float(mc["entropy"][0]), "gt_mask": gt_mask,
         "true_pct": float(b["severity"]) * 100, "head_pct": None, "rule_grade": None}
    if kind == "multitask":
        pm = mc["seg_logits"][0].argmax(0).cpu().numpy().astype(np.uint8)
        r["head_pct"] = float(mc["severity_mean"][0]) * 100
        heat, _ = cam(x, "grade", pred)
        sev = severity_from_mask(pm)
        r["rule_grade"] = grade_from_severity(sev, cfg["grading_rules"], grades)
    else:
        heat, _ = cam(x, "grade", grades.index("Reject"))
        pm = np.where(gt_mask > 0, 1, 0).astype(np.uint8)   # fruit region from GT (generous to baseline)
        pm[(gt_mask > 0) & (heat > 0.5)] = 3
        sev = severity_from_mask(pm)
    r.update(pm=pm, heat=heat, sev=sev)
    # review flag: the uncertainty rule (used for review_flag_rate) and the app's full rule set
    reasons = []
    if conf < icfg["review_confidence"]:
        reasons.append(f"low confidence ({conf:.0%})")
    if std > icfg["review_std"]:
        reasons.append(f"unstable under MC Dropout (std {std:.2f})")
    r["flag_uncertainty"] = bool(reasons)
    if r["rule_grade"] is not None and r["rule_grade"] != pred:
        reasons.append(f"classifier says {grades[pred]} but defect area implies {grades[r['rule_grade']]}")
    if (pm != 0).mean() < 0.03:
        reasons.append("no fruit found in the image")
    r["flag_reasons"] = reasons
    return r


def evaluate_model(ckpt, cfg, ds, device, n_samples=6, mc_passes=10, n_boot=1000, boot_seed=0,
                   seed=None, per_image=None):
    """seed: torch seed for the MC Dropout passes (None = not seeded).
    per_image(i, item, result): optional callback for every image (used by predict_split).
    Returns (metrics dict, sample list, meters for a paired bootstrap)."""
    if seed is not None:
        torch.manual_seed(seed)
    model, ck = load_checkpoint(ckpt, device)
    kind, grades = ck["kind"], cfg["grades"]
    cam = GradCAM(model)
    cls, seg = ClsMeter(len(grades)), SegMeter(len(cfg["seg_classes"]))
    mae_head, mae_mask = MAEMeter(), MAEMeter()
    rule_agree, flagged, flagged_wrong, wrong, flag_all, flag_all_wrong = 0, 0, 0, 0, 0, 0
    samples = []

    for i in range(len(ds)):
        b = ds[i]
        r = predict_image(model, kind, cam, b, cfg, mc_passes, device)
        pred, gt_mask, true_pct, pm = r["pred"], r["gt_mask"], r["true_pct"], r["pm"]
        cls.update([int(b["grade"])], [pred])

        is_flag = r["flag_uncertainty"]
        is_wrong = pred != int(b["grade"])
        flagged += is_flag
        wrong += is_wrong
        flagged_wrong += is_flag and is_wrong
        flag_all += bool(r["flag_reasons"])
        flag_all_wrong += bool(r["flag_reasons"]) and is_wrong

        if kind == "multitask":
            mae_head.update([true_pct], [r["head_pct"]])
            rule_agree += r["rule_grade"] == pred
        mae_mask.update([true_pct], [r["sev"]["total_pct"]])
        seg.update(gt_mask, pm)
        if per_image:
            per_image(i, b, r)

        if len(samples) < n_samples and i % max(1, len(ds) // n_samples) == 0:
            rgb = to_rgb(b["image"])
            samples.append({"rgb": rgb, "gt": gt_mask, "pred": pm, "cam": r["heat"],
                            "gt_grade": grades[int(b["grade"])], "pred_grade": grades[pred],
                            "true_pct": true_pct, "pred_pct": r["sev"]["total_pct"]})

    out = {"kind": kind, "checkpoint": str(ckpt), "n": len(ds), "mc_passes": mc_passes, "mc_seed": seed,
           "classification": cls.result(grades),
           "segmentation": seg.result(cfg["seg_classes"]),
           "severity_mae_from_mask": mae_mask.result(),
           "review_flag_rate": flagged / len(ds),
           "errors_caught_by_flag": (flagged_wrong / wrong) if wrong else None,
           "review_flag_rate_all_reasons": flag_all / len(ds),
           "errors_caught_by_flag_all_reasons": (flag_all_wrong / wrong) if wrong else None}
    if kind == "multitask":
        out["severity_mae_regression_head"] = mae_head.result()
        out["classifier_vs_mask_rule_agreement"] = rule_agree / len(ds)
    meters = (kind, cls, seg, mae_mask, mae_head)
    if n_boot:
        out["ci95"] = bootstrap_ci(len(ds), lambda idx: headline_metrics(kind, cfg, cls, seg, mae_mask, mae_head, idx),
                                   n_boot, boot_seed)
        out["bootstrap"] = {"resamples": n_boot, "seed": boot_seed, "unit": "image"}
    cam.remove()
    return out, samples, meters


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
    import textwrap
    cm = np.array(cm)
    title = "\n".join(textwrap.fill(line, 48) for line in title.split("\n"))   # long titles must not be cut off
    fig, ax = plt.subplots(figsize=(4.2, 3.8 + 0.18 * title.count("\n")))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=8 if "\n" in title else 10)
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
            # dark outline so the label is readable on white backgrounds too
            cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
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
        ("Flagged, all app reasons (+ rule mismatch)", lambda r: f(r.get("review_flag_rate_all_reasons"), True)),
        ("Wrong grades caught, all app reasons", lambda r: f(r.get("errors_caught_by_flag_all_reasons"), True)),
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
    ap.add_argument("--seed", type=int, default=None, help="torch seed for MC Dropout (default: not seeded)")
    ap.add_argument("--title", default=None, help="heading put above every table/figure (e.g. PRELIMINARY ...)")
    ap.add_argument("--image-size", type=int, default=None, help="evaluate at the size the model was trained on")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
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
        r, samples, _ = evaluate_model(ckpt, cfg, ds, device, mc_passes=args.mc_passes,
                                    n_boot=args.bootstrap, boot_seed=args.boot_seed,
                                       seed=args.seed)
        results.append(r)
        plot_confusion(r["classification"]["confusion_matrix"], cfg["grades"],
                       (f"{args.title}\n" if args.title else "")
                       + f"{r['kind']} — acc {r['classification']['accuracy']:.1%}",
                       out / f"confusion_{r['kind']}.png")
        save_sample_grid(samples, out / f"samples_{r['kind']}.png", r["kind"])
        print(json.dumps({k: v for k, v in r.items() if k not in ("classification",)}, indent=1)[:1500])

    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    table = comparison_table(results)
    head = f"# {args.title}\n\n" if args.title else ""
    (out / "comparison.md").write_text(f"{head}## Test results ({len(ds)} images, split={args.split})\n\n{table}\n",
                                       encoding="utf-8")
    print("\n" + table)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
