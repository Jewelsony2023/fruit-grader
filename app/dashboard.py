"""
Data for the presentation dashboard (/dashboard). Every number is READ from a file
in the report folder (or a file it names); this module only regroups and counts.
The browser only draws what it gets from here.

Files used (FRUIT_DASHBOARD_DIR, default results/preliminary):
    metrics.json, paired_bootstrap.json, model_info.json, predictions.csv   (fruitgrade.predict_split)
    <run_dir>/history.csv for each model                                    (fruitgrade.train)
    results/data_audit/dataset_audit.json                                   (fruitgrade.data.dedupe + label_filter)
"""
import csv
import json
from pathlib import Path

from fruitgrade.config import PROJECT_ROOT

BANNER = "PRELIMINARY — test ground truth = auto-generated pseudo-masks"
MODELS = ("multitask", "baseline")
GRADES = ("A", "B", "C", "Reject")
AUDIT = "results/data_audit/dataset_audit.json"
NUM = {"correct", "flagged", "flagged_uncertainty"}


def _num(row):
    out = {}
    for k, v in row.items():
        if k in NUM:
            out[k] = int(v)
        elif k in ("model", "image", "fruit", "fresh_rotten", "true_grade", "pred_grade", "flag_reasons",
                   "rule_grade", "thumb"):
            out[k] = v
        else:
            out[k] = float(v) if v != "" else None
    out["id"] = Path(row["image"]).stem
    return out


def read_predictions(path):
    return [_num(r) for r in csv.DictReader(open(path, encoding="utf-8"))]


def reliability(rows, n_bins=10):
    """Equal-width confidence bins on [0, 1]: count, mean confidence, accuracy. ECE = sum n/N |acc - conf|."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [r for r in rows if lo <= r["confidence"] < hi or (b == n_bins - 1 and r["confidence"] == 1.0)]
        bins.append({"lo": lo, "hi": hi, "n": len(sel),
                     "accuracy": sum(r["correct"] for r in sel) / len(sel) if sel else None,
                     "mean_conf": sum(r["confidence"] for r in sel) / len(sel) if sel else None})
    ece = sum(b["n"] / len(rows) * abs(b["accuracy"] - b["mean_conf"]) for b in bins if b["n"])
    return {"bins": bins, "ece": ece}


def threshold_curve(rows, thresholds=None):
    """Accept only images with confidence >= t: what share is kept and how accurate it is."""
    thresholds = thresholds or [round(0.25 + 0.05 * i, 2) for i in range(15)]
    out = []
    for t in thresholds:
        kept = [r for r in rows if r["confidence"] >= t]
        out.append({"t": t, "coverage": len(kept) / len(rows),
                    "accuracy": sum(r["correct"] for r in kept) / len(kept) if kept else None, "n": len(kept)})
    return out


def flag_summary(rows):
    wrong = [r for r in rows if not r["correct"]]
    out = {}
    for key, name in (("flagged_uncertainty", "Uncertainty only (confidence < threshold or MC std > threshold)"),
                      ("flagged", "All app reasons (+ classifier vs mask-rule mismatch, no fruit)")):
        fl = [r for r in rows if r[key]]
        ok = [r for r in rows if not r[key]]
        out[key] = {"rule": name, "n": len(rows), "flagged": len(fl), "flag_rate": len(fl) / len(rows),
                    "errors": len(wrong), "errors_caught": sum(1 for r in wrong if r[key]),
                    "errors_caught_rate": sum(1 for r in wrong if r[key]) / len(wrong) if wrong else None,
                    "accuracy_unflagged": sum(r["correct"] for r in ok) / len(ok) if ok else None}
    return out


def severity_block(rows, kind):
    out = {"points": [[r["true_severity"], r["pred_severity_mask"], r["pred_severity_head"], r["id"]] for r in rows]}
    edges = list(range(-50, 55, 5))           # error = predicted - true, % points, 5-point bins
    for src in ("mask", "head"):
        errs = [r[f"pred_severity_{src}"] - r["true_severity"] for r in rows if r[f"pred_severity_{src}"] is not None]
        if not errs:
            continue
        counts = [0] * (len(edges) + 1)       # first = below -50, last = 50 and above
        for e in errs:
            k = 0 if e < edges[0] else len(edges) if e >= edges[-1] else 1 + next(
                i for i in range(len(edges) - 1) if edges[i] <= e < edges[i + 1])
            counts[k] += 1
        out[f"hist_{src}"] = {"edges": edges, "counts": counts}
        out[f"mae_by_grade_{src}"] = {
            g: ({"mae": sum(abs(r[f"pred_severity_{src}"] - r["true_severity"]) for r in sel) / len(sel), "n": len(sel)}
                if (sel := [r for r in rows if r["true_grade"] == g]) else None) for g in GRADES}
    return out


def worst_errors(rows, k=12):
    wrong = sorted((r for r in rows if not r["correct"]), key=lambda r: -r["confidence"])[:k]
    sev = sorted(rows, key=lambda r: -abs(r["pred_severity_mask"] - r["true_severity"]))[:k]
    return {"confident_wrong": [r["id"] for r in wrong], "severity": [r["id"] for r in sev]}


def read_history(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = rows[0].keys() if rows else []
    return {c: [(float(r[c]) if r[c] not in ("", "True", "False") else (r[c] == "True" if r[c] else None))
                for r in rows] for c in cols}


def data_block(audit_path):
    a = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    vers = a["versions_per_original_hist"]
    return {"files_total": a["files_total"], "originals": a["originals"],
            "versions_per_original": int(next(iter(vers))) if len(vers) == 1 else None,
            "augmentation_type_counts": a["augmentation_type_counts"],
            "kaggle_split_leak_originals": a["kaggle_split_leak_originals"],
            "originals_used": a["originals_used"], "candidate_pairs": a["candidate_pairs_within_max_dist"],
            "merged_pairs": len(a["merged_pairs"]), "label_conflicts": len(a["label_conflict"]),
            "groups": a["groups"], "seed": a["seed"], "split_images": a["split_images"],
            "split_groups": a["split_groups"], "split_by_class": a["split_by_class"],
            "label_filter": {k: a["label_consistency_filter"][k] for k in ("rule", "excluded", "by_reason_split")},
            "grade_by_split": a["label_consistency_filter"]["remaining_split_grade"],
            "tuning_sample": a.get("pseudo_mask_tuning_sample")}


def build(rep):
    """Everything the dashboard shows, from the files in report folder rep."""
    rep = Path(rep)
    metrics = {m["kind"]: m for m in json.loads((rep / "metrics.json").read_text())}
    paired = json.loads((rep / "paired_bootstrap.json").read_text())
    info = json.loads((rep / "model_info.json").read_text())
    preds = read_predictions(rep / "predictions.csv")
    by = {k: [r for r in preds if r["model"] == k] for k in MODELS}
    rel = lambda p: Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()  # noqa: E731
    out = {"banner": BANNER, "report_dir": rel(rep), "metrics": metrics, "paired": paired, "info": info,
           "predictions": {k: {r["id"]: r for r in v} for k, v in by.items()},
           "sources": {"metrics": rel(rep / "metrics.json"), "paired": rel(rep / "paired_bootstrap.json"),
                       "info": rel(rep / "model_info.json"), "predictions": rel(rep / "predictions.csv"),
                       "audit": AUDIT},
           "confusion": {}, "seg_images": {}, "severity": {}, "confidence": {}, "errors": {}, "history": {},
           "warnings": []}
    for k in MODELS:
        rows = by[k]
        cells = {}
        for r in rows:
            cells.setdefault(f"{GRADES.index(r['true_grade'])},{GRADES.index(r['pred_grade'])}", []).append(r["id"])
        out["confusion"][k] = cells
        cm = metrics[k]["classification"]["confusion_matrix"]
        if any(len(cells.get(f"{i},{j}", [])) != cm[i][j] for i in range(4) for j in range(4)):
            out["warnings"].append(f"{k}: predictions.csv does not match the confusion matrix in metrics.json")
        out["seg_images"][k] = {d: sum(1 for r in rows if r[f"true_{d}_pct"] > 0) for d in ("bruise", "rot", "mechanical")}
        out["severity"][k] = severity_block(rows, k)
        out["confidence"][k] = {"reliability": reliability(rows), "threshold": threshold_curve(rows),
                                "flags": flag_summary(rows)}
        out["errors"][k] = worst_errors(rows)
        hist = PROJECT_ROOT / info["models"][k]["run_dir"] / "history.csv"
        if hist.exists():
            out["history"][k] = read_history(hist)
            out["sources"][f"history_{k}"] = rel(hist)
    out["data"] = data_block(PROJECT_ROOT / AUDIT)
    return out
