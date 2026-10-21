"""
Metrics (accumulated over a whole split, not averaged per batch):

  Classification : accuracy, macro-F1, per-class recall, confusion matrix
  Segmentation   : IoU and Dice per class, mean over DEFECT classes,
                   plus "any-defect" binary IoU/Dice (defect vs not)
  Severity       : MAE in percentage points (|pred% - true%|)
  Comparison     : paired bootstrap of the difference between two models

IoU  = TP / (TP + FP + FN)
Dice = 2TP / (2TP + FP + FN)

Each meter also keeps one record per update() call, so result(idx=...) can
recompute the metric on a resample of those calls. evaluate.py calls update()
once per image, which gives bootstrap confidence intervals "by image".
"""
import numpy as np


class ClsMeter:
    def __init__(self, n):
        self.n = n
        self.cm = np.zeros((n, n), np.int64)
        self.records = []

    def update(self, y_true, y_pred):
        cm = np.zeros((self.n, self.n), np.int64)
        for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
            cm[t, p] += 1
        self.cm += cm
        self.records.append(cm)

    def result(self, names, idx=None):
        cm = self.cm if idx is None else np.stack(self.records)[idx].sum(0)
        tp = np.diag(cm).astype(float)
        recall = tp / np.maximum(cm.sum(1), 1)
        prec = tp / np.maximum(cm.sum(0), 1)
        f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-9)
        present = cm.sum(1) > 0
        return {
            "accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "per_class_recall": {n: float(r) for n, r in zip(names, recall)},
            "per_class": {n: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                          for n, p, r, f, s in zip(names, prec, recall, f1, cm.sum(1))},
            "confusion_matrix": cm.tolist(),
        }


class SegMeter:
    def __init__(self, n_classes, defect_ids=(2, 3, 4)):
        self.n = n_classes
        self.defect_ids = list(defect_ids)
        self.tp = np.zeros(n_classes)
        self.fp = np.zeros(n_classes)
        self.fn = np.zeros(n_classes)
        self.bin = np.zeros(3)  # tp, fp, fn for any-defect
        self.records = []       # per update: (tp, fp, fn per class, bin)

    def update(self, gt, pred):
        gt, pred = np.asarray(gt), np.asarray(pred)
        rec = np.zeros((3, self.n))
        for c in range(self.n):
            g, p = gt == c, pred == c
            rec[:, c] = [np.logical_and(g, p).sum(), np.logical_and(~g, p).sum(),
                         np.logical_and(g, ~p).sum()]
        g, p = np.isin(gt, self.defect_ids), np.isin(pred, self.defect_ids)
        b = np.array([np.logical_and(g, p).sum(), np.logical_and(~g, p).sum(),
                      np.logical_and(g, ~p).sum()], float)
        self.tp += rec[0]
        self.fp += rec[1]
        self.fn += rec[2]
        self.bin += b
        self.records.append((rec, b))

    def result(self, names, idx=None):
        if idx is None:
            tp_c, fp_c, fn_c, bin_ = self.tp, self.fp, self.fn, self.bin
        else:
            tp_c, fp_c, fn_c = np.stack([r[0] for r in self.records])[idx].sum(0)
            bin_ = np.stack([r[1] for r in self.records])[idx].sum(0)
        iou = tp_c / np.maximum(tp_c + fp_c + fn_c, 1)
        dice = 2 * tp_c / np.maximum(2 * tp_c + fp_c + fn_c, 1)
        present = [c for c in self.defect_ids if tp_c[c] + fn_c[c] > 0]
        tp, fp, fn = bin_
        return {
            "iou": {n: float(v) for n, v in zip(names, iou)},
            "dice": {n: float(v) for n, v in zip(names, dice)},
            "mean_defect_iou": float(np.mean(iou[present])) if present else 0.0,
            "mean_defect_dice": float(np.mean(dice[present])) if present else 0.0,
            "any_defect_iou": float(tp / max(tp + fp + fn, 1)),
            "any_defect_dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "gt_pixels": {n: int(v) for n, v in zip(names, tp_c + fn_c)},
        }


class MAEMeter:
    def __init__(self):
        self.err, self.n = 0.0, 0
        self.records = []       # per update: (sum of |error|, count)

    def update(self, true_pct, pred_pct):
        t, p = np.asarray(true_pct, float), np.asarray(pred_pct, float)
        e = np.abs(t - p).sum()
        self.err += e
        self.n += t.size
        self.records.append((e, t.size))

    def result(self, idx=None):
        if idx is None:
            return float(self.err / max(self.n, 1))
        r = np.array(self.records)[idx]
        return float(r[:, 0].sum() / max(r[:, 1].sum(), 1))


def bootstrap_ci(n_items, stat_fn, n_boot=1000, seed=0, level=0.95):
    """Percentile bootstrap CI. stat_fn(idx) -> dict of metric -> value, computed on the
    items idx (drawn with replacement). Returns {metric: [low, high]}."""
    rng = np.random.default_rng(seed)
    draws = [stat_fn(rng.integers(0, n_items, n_items)) for _ in range(n_boot)]
    a = (1 - level) / 2 * 100
    return {k: [float(np.percentile([d[k] for d in draws], a)),
                float(np.percentile([d[k] for d in draws], 100 - a))] for k in draws[0]}


def paired_bootstrap_diff(n_items, stat_a, stat_b, n_boot=1000, seed=0, level=0.95):
    """Paired percentile bootstrap of stat_a - stat_b. Both models are scored on the SAME
    resampled items in every draw, so the per-image correlation between them is kept.
    stat_x(idx) -> dict of metric -> value (idx=None = all items).
    Returns {metric: {"diff", "ci95": [low, high], "excludes_zero"}}."""
    rng = np.random.default_rng(seed)
    full_a, full_b = stat_a(None), stat_b(None)
    keys = [k for k in full_a if k in full_b]
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        a, b = stat_a(idx), stat_b(idx)
        draws.append([a[k] - b[k] for k in keys])
    draws = np.array(draws)
    q = (1 - level) / 2 * 100
    out = {}
    for j, k in enumerate(keys):
        lo, hi = float(np.percentile(draws[:, j], q)), float(np.percentile(draws[:, j], 100 - q))
        out[k] = {"diff": float(full_a[k] - full_b[k]), "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
    return out
