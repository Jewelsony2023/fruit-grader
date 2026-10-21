"""
Per-image predictions of both models on one split (default: test), for the dashboard.

    python -m fruitgrade.predict_split \
        --multitask runs/prelim_multitask_efficientnet_b0/best.pt \
        --baseline  runs/prelim_baseline_efficientnet_b0/best.pt \
        --out results/preliminary --title "PRELIMINARY ..."

Uses evaluate.evaluate_model (same per-image code as metrics.json), MC Dropout with
20 passes and a fixed torch seed, so the CSV and the metrics always agree.

Outputs (in --out):
    predictions.csv        one row per image x model (columns: PRED_COLUMNS)
    thumbs/<model>/<id>.jpg  input | true mask | predicted mask | Grad-CAM
    metrics.json           as evaluate.py, plus per-class precision/recall/F1/support
    paired_bootstrap.json  multitask - baseline, SAME resampled images for both models
    model_info.json        backbone, parameter count, epochs, best epoch, train time, GPU
    comparison.md, confusion_<model>.png, samples_<model>.png
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fruitgrade.config import PROJECT_ROOT, load_config
from fruitgrade.data.dataset import FruitDataset
from fruitgrade.evaluate import (comparison_table, evaluate_model, headline_metrics, plot_confusion,
                                 save_sample_grid, to_rgb)
from fruitgrade.grading import colourise_mask, severity_from_mask
from fruitgrade.inference import heatmap_overlay
from fruitgrade.metrics import paired_bootstrap_diff
from fruitgrade.models import load_checkpoint
from fruitgrade.train import get_device

DEFECTS = ("bruise", "rot", "mechanical")
PRED_COLUMNS = (["model", "image", "fruit", "fresh_rotten", "true_grade", "pred_grade", "correct",
                 "p_A", "p_B", "p_C", "p_Reject", "confidence", "mc_std", "entropy",
                 "flagged", "flagged_uncertainty", "flag_reasons", "rule_grade",
                 "true_severity", "pred_severity_mask", "pred_severity_head"]
                + [f"{w}_{d}_pct" for d in DEFECTS for w in ("true", "pred")]
                + ["defect_iou", "defect_dice", "thumb"])
# metrics compared by the paired bootstrap (multitask - baseline)
PAIRED = ["accuracy", "macro_f1", "any_defect_dice", "severity_mae_from_mask"]
HIGHER_IS_BETTER = {"accuracy": True, "macro_f1": True, "any_defect_dice": True,
                    "severity_mae_from_mask": False, "severity_mae_head_vs_baseline_mask": False}


def any_defect_overlap(gt, pm):
    """Per-image any-defect IoU and Dice. None when neither mask has a defect pixel (undefined)."""
    g, p = gt >= 2, pm >= 2
    tp, fp, fn = (g & p).sum(), (~g & p).sum(), (g & ~p).sum()
    if tp + fp + fn == 0:
        return None, None
    return float(tp / (tp + fp + fn)), float(2 * tp / (2 * tp + fp + fn))


def thumbnail(b, r, size):
    rgb = to_rgb(b["image"])
    panels = [rgb, colourise_mask(r["gt_mask"], True), colourise_mask(r["pm"], True), heatmap_overlay(rgb, r["heat"])]
    panels = [cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) for p in panels]
    return cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR)


def make_row(kind, item, row, r, grades, thumb):
    true_sev = severity_from_mask(r["gt_mask"])
    iou, dice = any_defect_overlap(r["gt_mask"], r["pm"])
    f = lambda v: "" if v is None else round(float(v), 6)  # noqa: E731
    out = {"model": kind, "image": row["image"], "fruit": row["fruit"],
           "fresh_rotten": row["source"].replace("kaggle_", ""),
           "true_grade": grades[int(item["grade"])], "pred_grade": grades[r["pred"]],
           "correct": int(r["pred"] == int(item["grade"])),
           **{f"p_{g}": f(p) for g, p in zip(grades, r["probs"])},
           "confidence": f(r["conf"]), "mc_std": f(r["std"]), "entropy": f(r["entropy"]),
           "flagged": int(bool(r["flag_reasons"])), "flagged_uncertainty": int(r["flag_uncertainty"]),
           "flag_reasons": "; ".join(r["flag_reasons"]),
           "rule_grade": "" if r["rule_grade"] is None else grades[r["rule_grade"]],
           "true_severity": f(r["true_pct"]), "pred_severity_mask": f(r["sev"]["total_pct"]),
           "pred_severity_head": f(r["head_pct"]),
           "defect_iou": f(iou), "defect_dice": f(dice), "thumb": thumb}
    for d in DEFECTS:
        out[f"true_{d}_pct"] = f(true_sev[f"{d}_pct"])
        out[f"pred_{d}_pct"] = f(r["sev"][f"{d}_pct"])
    return out


def model_info(kind, ckpt, out_dir):
    """Facts about one trained model, each read from the checkpoint / run folder / run_settings.json."""
    model, ck = load_checkpoint(ckpt, "cpu")
    run = Path(ckpt).parent
    hist = list(csv.DictReader(open(run / "history.csv", encoding="utf-8")))
    info = {"kind": kind, "checkpoint": Path(ckpt).as_posix(), "run_dir": run.as_posix(),
            "backbone": ck["config"]["model"]["backbone"], "pretrained": ck["config"]["model"]["pretrained"],
            "image_size": ck["config"]["data"]["image_size"],
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "epochs_run": len(hist), "best_epoch": ck.get("epoch"),
            "train_seconds_sum_of_epochs": round(sum(float(h["sec"]) for h in hist), 1),
            "frozen_backbone_epochs": sum(h.get("frozen") == "True" for h in hist),
            "sources": {"parameters/best_epoch/backbone": Path(ckpt).as_posix(),
                        "epochs/train_seconds": (run / "history.csv").as_posix()}}
    if (run / "summary.json").exists():
        info["best_val_score"] = json.loads((run / "summary.json").read_text())["best_score"]
    rs = Path(out_dir) / "run_settings.json"
    if rs.exists():
        s = json.loads(rs.read_text(encoding="utf-8-sig")).get(kind, {})
        info.update(train_minutes=s.get("minutes"), batch_size=s.get("batch"))
        info["sources"]["train_minutes/batch_size"] = rs.as_posix()
        log = PROJECT_ROOT / f"logs/train_prelim_{kind}_bs{s.get('batch')}_{s.get('image_size')}.out.log"
        if log.exists():
            first = log.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            info["train_device"] = first.split("|")[0].replace("Device:", "").strip()
            info["sources"]["train_device"] = log.relative_to(PROJECT_ROOT).as_posix()
    if torch.cuda.is_available():
        info["gpu_this_machine"] = torch.cuda.get_device_name(0)
    return info


def paired_section(paired, n_boot):
    names = {"accuracy": "Grade accuracy", "macro_f1": "Grade macro-F1", "any_defect_dice": "Any-defect Dice",
             "severity_mae_from_mask": "Severity MAE (mask, % pts)",
             "severity_mae_head_vs_baseline_mask": "Severity MAE (multitask head vs baseline mask, % pts)"}
    lines = [f"## Paired bootstrap: multitask - baseline ({n_boot} resamples, same images for both models)", "",
             "| Metric | Difference | 95% CI | CI excludes 0? |", "|---|---|---|---|"]
    for k, v in paired.items():
        lo, hi = v["ci95"]
        lines.append(f"| {names.get(k, k)} | {v['diff']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{'yes' if v['excludes_zero'] else 'no'} |")
    lines += ["", "For MAE a negative difference means the multitask model is better."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multitask", default="runs/prelim_multitask_efficientnet_b0/best.pt")
    ap.add_argument("--baseline", default="runs/prelim_baseline_efficientnet_b0/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/preliminary")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mc-passes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the MC Dropout passes")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--thumb-size", type=int, default=160, help="side of each of the 4 thumbnail panels")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sizes = {load_checkpoint(c, "cpu")[1]["config"]["data"]["image_size"] for c in (args.multitask, args.baseline)}
    if len(sizes) != 1:
        raise SystemExit(f"models were trained at different image sizes {sizes}; the paired comparison needs one")
    cfg["data"]["image_size"] = sizes.pop()
    device = get_device()
    ds = FruitDataset(cfg["data"]["root"], args.split, cfg["grades"], cfg["data"]["image_size"], train=False)
    out = Path(args.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    print(f"Predicting {len(ds)} '{args.split}' images, MC Dropout {args.mc_passes} passes, seed {args.seed}")

    results, rows, meters, infos = [], [], {}, {}
    for kind, ckpt in (("multitask", args.multitask), ("baseline", args.baseline)):
        tdir = out / "thumbs" / kind
        tdir.mkdir(exist_ok=True)
        model_rows = []

        def per_image(i, item, r, kind=kind, tdir=tdir, model_rows=model_rows):
            name = Path(item["name"]).stem
            cv2.imwrite(str(tdir / f"{name}.jpg"), thumbnail(item, r, args.thumb_size), [cv2.IMWRITE_JPEG_QUALITY, 85])
            model_rows.append(make_row(kind, item, ds.rows[i], r, cfg["grades"], f"thumbs/{kind}/{name}.jpg"))

        r, samples, m = evaluate_model(ckpt, cfg, ds, device, mc_passes=args.mc_passes, n_boot=args.bootstrap,
                                       boot_seed=args.boot_seed, seed=args.seed, per_image=per_image)
        if r["kind"] != kind:
            raise SystemExit(f"{ckpt} is a '{r['kind']}' checkpoint, expected '{kind}'")
        results.append(r)
        rows += model_rows
        meters[kind] = m
        infos[kind] = model_info(kind, ckpt, out)
        infos[kind]["mc_passes"], infos[kind]["mc_seed"] = args.mc_passes, args.seed
        plot_confusion(r["classification"]["confusion_matrix"], cfg["grades"],
                       (f"{args.title}\n" if args.title else "") + f"{kind} — acc {r['classification']['accuracy']:.1%}",
                       out / f"confusion_{kind}.png")
        save_sample_grid(samples, out / f"samples_{kind}.png", kind)
        print(f"{kind}: accuracy {r['classification']['accuracy']:.4f}, macro-F1 {r['classification']['macro_f1']:.4f}")

    # the paired bootstrap is only valid if both models saw the same images in the same order
    imgs = {k: [x["image"] for x in rows if x["model"] == k] for k in meters}
    assert imgs["multitask"] == imgs["baseline"], "image order differs between models"

    def stat(kind, other_sev=False):
        _, cls, seg, mae_mask, mae_head = meters[kind]

        def fn(idx):
            h = headline_metrics(kind, cfg, cls, seg, mae_mask, mae_head, idx)
            d = {k: h[k] for k in PAIRED}
            d["severity_mae_head_vs_baseline_mask"] = h["severity_mae_regression_head"] if kind == "multitask" \
                else h["severity_mae_from_mask"]
            return d
        return fn

    paired = paired_bootstrap_diff(len(ds), stat("multitask"), stat("baseline"), args.bootstrap, args.boot_seed)
    for k, v in paired.items():
        better = (v["diff"] > 0) == HIGHER_IS_BETTER[k]
        v["higher_is_better"] = HIGHER_IS_BETTER[k]
        v["verdict"] = ("multitask better" if better else "baseline better") if v["excludes_zero"] \
            else "no clear difference (CI includes 0)"

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, PRED_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    json.dump({"what": "paired percentile bootstrap of multitask - baseline", "split": args.split, "n": len(ds),
               "resamples": args.bootstrap, "seed": args.boot_seed, "unit": "image", "metrics": paired},
              open(out / "paired_bootstrap.json", "w"), indent=2)
    json.dump({"split": args.split, "n_images": len(ds), "models": infos}, open(out / "model_info.json", "w"), indent=2)
    head = f"# {args.title}\n\n" if args.title else ""
    (out / "comparison.md").write_text(
        f"{head}## Test results ({len(ds)} images, split={args.split}, MC Dropout {args.mc_passes} passes, "
        f"seed {args.seed})\n\n{comparison_table(results)}\n\n{paired_section(paired, args.bootstrap)}\n",
        encoding="utf-8")
    print(paired_section(paired, args.bootstrap))
    print(f"{len(rows)} rows -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
