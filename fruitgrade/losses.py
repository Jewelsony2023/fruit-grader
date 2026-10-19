"""
Multi-task loss
    L = w_cls * CE(grade)  +  w_seg * (CE(mask) + Dice(mask))  +  w_sev * SmoothL1(severity)

* grade CE uses class weights (inverse frequency) -> handles imbalance
  (real datasets have far more A/Reject than B/C).
* Dice loss is added to segmentation because defects are SMALL compared to
  background/healthy pixels; pixel CE alone would ignore them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, ignore_bg=True, eps=1.0):
    n = logits.shape[1]
    prob = logits.softmax(1)
    onehot = F.one_hot(target, n).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    union = prob.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    start = 1 if ignore_bg else 0
    return 1 - dice[start:].mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_sev=2.0, grade_weights=None, seg_weights=None):
        super().__init__()
        self.w_cls, self.w_seg, self.w_sev = w_cls, w_seg, w_sev
        self.ce_grade = nn.CrossEntropyLoss(weight=grade_weights)
        self.ce_seg = nn.CrossEntropyLoss(weight=seg_weights)

    def forward(self, out, batch):
        parts = {"cls": self.ce_grade(out["grade"], batch["grade"])}
        total = self.w_cls * parts["cls"]
        if "seg" in out:
            parts["seg"] = self.ce_seg(out["seg"], batch["mask"]) + soft_dice_loss(out["seg"], batch["mask"])
            total = total + self.w_seg * parts["seg"]
        if "severity" in out:
            parts["sev"] = F.smooth_l1_loss(out["severity"], batch["severity"], beta=0.05)
            total = total + self.w_sev * parts["sev"]
        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}
