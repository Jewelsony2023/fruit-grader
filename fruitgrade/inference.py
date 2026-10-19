"""
Predictor: one image in -> everything the UI shows out.

    from fruitgrade.inference import Predictor
    p = Predictor("runs/multitask_efficientnet_b0/best.pt")
    result = p.predict(rgb_numpy_image)

CLI:
    python -m fruitgrade.inference --ckpt runs/.../best.pt --image apple.jpg --save out.png
"""
import argparse

import cv2
import numpy as np
import torch

from fruitgrade.data.dataset import preprocess
from fruitgrade.data.pseudo_masks import segment_fruit
from fruitgrade.explain import GradCAM, mc_dropout
from fruitgrade.grading import (BACKGROUND, BRUISE, colourise_mask, grade_from_severity,
                                severity_from_mask)
from fruitgrade.models import load_checkpoint
from fruitgrade.train import get_device


def heatmap_overlay(rgb, cam, alpha=0.45):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def mask_overlay(rgb, mask, alpha=0.6):
    out = rgb.copy()
    col = colourise_mask(mask)
    sel = mask >= BRUISE
    out[sel] = (rgb[sel] * (1 - alpha) + col[sel] * alpha).astype(np.uint8)
    # outline each defect region so small ones stay visible
    for cid in np.unique(mask[sel]) if sel.any() else []:
        cnts, _ = cv2.findContours((mask == cid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = tuple(int(v) for v in colourise_mask(np.array([[cid]]))[0, 0])
        cv2.drawContours(out, cnts, -1, c, max(1, rgb.shape[0] // 250))
    return out


class Predictor:
    def __init__(self, ckpt_path, device="auto"):
        self.device = get_device(device)
        self.model, ck = load_checkpoint(ckpt_path, self.device)
        self.kind = ck["kind"]
        self.cfg = ck["config"]
        self.grades = self.cfg["grades"]
        self.size = self.cfg["data"]["image_size"]
        self.icfg = self.cfg["inference"]
        self.cam = GradCAM(self.model)
        self.meta = {"kind": self.kind, "backbone": self.cfg["model"]["backbone"],
                     "epoch": ck.get("epoch"), "val": ck.get("val")}

    def _tensor(self, rgb):
        return preprocess(rgb, self.size).to(self.device)

    def gradcam(self, rgb, target=None, kind="grade"):
        cam, _ = self.cam(self._tensor(rgb), target=kind, index=target)
        return cam

    def predict(self, rgb: np.ndarray, max_side=512, passes=None):
        h, w = rgb.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        H, W = rgb.shape[:2]
        x = self._tensor(rgb)
        passes = passes or self.icfg["mc_dropout_passes"]

        # 1) MC Dropout: grade probabilities + uncertainty (+ severity, + mask)
        mc = mc_dropout(self.model, x, passes)
        probs = mc["probs"][0].cpu().numpy()
        top = int(probs.argmax())
        conf = float(probs[top])
        top_std = float(mc["top_std"][0])
        entropy = float(mc["entropy"][0])

        # 2) Grad-CAM for the predicted grade (and for "Reject" = defect evidence)
        cam_pred, _ = self.cam(x, "grade", top)
        cam_pred = cv2.resize(cam_pred, (W, H))

        res = {
            "model": self.meta["kind"], "backbone": self.meta["backbone"],
            "grade": self.grades[top],
            "grade_probs": {g: float(p) for g, p in zip(self.grades, probs)},
            "confidence": conf, "mc_std": top_std, "entropy": entropy, "mc_passes": passes,
        }

        if mc["seg_logits"] is not None:
            # 3) segmentation -> defect mask -> severity % per defect type
            seg = mc["seg_logits"][0].softmax(0)
            seg = torch.nn.functional.interpolate(seg[None], size=(H, W), mode="bilinear",
                                                  align_corners=False)[0]
            mask = seg.argmax(0).cpu().numpy().astype(np.uint8)
            sev = severity_from_mask(mask)
            res["severity_score"] = float(mc["severity_mean"][0]) * 100      # regression head
            res["severity_std"] = float(mc["severity_std"][0]) * 100
            res["defect_area_pct"] = {k: round(sev[f"{k}_pct"], 2) for k in ("bruise", "rot", "mechanical")}
            res["defect_area_pct"]["total"] = round(sev["total_pct"], 2)
            rule = grade_from_severity(sev, self.cfg["grading_rules"], self.grades)
            res["rule_grade"] = self.grades[rule]
        else:
            # baseline: no mask -> weak proxy = Grad-CAM of "Reject" thresholded inside the fruit
            cam_rej, _ = self.cam(x, "grade", self.grades.index("Reject"))
            cam_rej = cv2.resize(cam_rej, (W, H))
            fruit = segment_fruit(rgb)
            mask = np.zeros((H, W), np.uint8)
            mask[fruit] = 1
            mask[fruit & (cam_rej > 0.5)] = 3
            sev = severity_from_mask(mask)
            res["severity_score"] = round(sev["total_pct"], 2)
            res["severity_std"] = None
            res["defect_area_pct"] = {"total": round(sev["total_pct"], 2)}
            res["rule_grade"] = None
            res["note"] = "Baseline model: no segmentation head. Severity is a Grad-CAM proxy."

        # 4) manual-review flag with human-readable reasons
        reasons = []
        if conf < self.icfg["review_confidence"]:
            reasons.append(f"low confidence ({conf:.0%})")
        if top_std > self.icfg["review_std"]:
            reasons.append(f"unstable under MC Dropout (std {top_std:.2f})")
        if res.get("rule_grade") and res["rule_grade"] != res["grade"]:
            reasons.append(f"classifier says {res['grade']} but defect area implies {res['rule_grade']}")
        if (mask != BACKGROUND).mean() < 0.03:
            reasons.append("no fruit found in the image")
        res["needs_review"] = bool(reasons)
        res["review_reasons"] = reasons

        res["images"] = {
            "input": rgb,
            "defect_overlay": mask_overlay(rgb, mask),
            "mask": colourise_mask(mask, show_healthy=True),
            "gradcam": heatmap_overlay(rgb, cam_pred),
        }
        res["_mask"] = mask
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default=None, help="save a 4-panel figure here")
    args = ap.parse_args()
    p = Predictor(args.ckpt)
    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    r = p.predict(rgb)
    for k, v in r.items():
        if not k.startswith("_") and k != "images":
            print(f"{k:16s}: {v}")
    if args.save:
        im = r["images"]
        panel = np.hstack([im["input"], im["defect_overlay"], im["mask"], im["gradcam"]])
        cv2.imwrite(args.save, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print("saved", args.save)


if __name__ == "__main__":
    main()
