"""
Metrics (accumulated over a whole split, not averaged per batch):

  Classification : accuracy, macro-F1, per-class recall, confusion matrix
  Segmentation   : IoU and Dice per class, mean over DEFECT classes,
                   plus "any-defect" binary IoU/Dice (defect vs not)
  Severity       : MAE in percentage points (|pred% - true%|)

IoU  = TP / (TP + FP + FN)
Dice = 2TP / (2TP + FP + FN)
"""
import numpy as np


class ClsMeter:
    def __init__(self, n):
        self.cm = np.zeros((n, n), np.int64)

    def update(self, y_true, y_pred):
        for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
            self.cm[t, p] += 1

    def result(self, names):
        cm = self.cm
        tp = np.diag(cm).astype(float)
        recall = tp / np.maximum(cm.sum(1), 1)
        prec = tp / np.maximum(cm.sum(0), 1)
        f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-9)
        present = cm.sum(1) > 0
        return {
            "accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "per_class_recall": {n: float(r) for n, r in zip(names, recall)},
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

    def update(self, gt, pred):
        gt, pred = np.asarray(gt), np.asarray(pred)
        for c in range(self.n):
            g, p = gt == c, pred == c
            self.tp[c] += np.logical_and(g, p).sum()
            self.fp[c] += np.logical_and(~g, p).sum()
            self.fn[c] += np.logical_and(g, ~p).sum()
        g, p = np.isin(gt, self.defect_ids), np.isin(pred, self.defect_ids)
        self.bin += [np.logical_and(g, p).sum(), np.logical_and(~g, p).sum(),
                     np.logical_and(g, ~p).sum()]

    def result(self, names):
        iou = self.tp / np.maximum(self.tp + self.fp + self.fn, 1)
        dice = 2 * self.tp / np.maximum(2 * self.tp + self.fp + self.fn, 1)
        present = [c for c in self.defect_ids if self.tp[c] + self.fn[c] > 0]
        tp, fp, fn = self.bin
        return {
            "iou": {n: float(v) for n, v in zip(names, iou)},
            "dice": {n: float(v) for n, v in zip(names, dice)},
            "mean_defect_iou": float(np.mean(iou[present])) if present else 0.0,
            "mean_defect_dice": float(np.mean(dice[present])) if present else 0.0,
            "any_defect_iou": float(tp / max(tp + fp + fn, 1)),
            "any_defect_dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        }


class MAEMeter:
    def __init__(self):
        self.err, self.n = 0.0, 0

    def update(self, true_pct, pred_pct):
        t, p = np.asarray(true_pct, float), np.asarray(pred_pct, float)
        self.err += np.abs(t - p).sum()
        self.n += t.size

    def result(self):
        return float(self.err / max(self.n, 1))
