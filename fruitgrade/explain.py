"""
Explainability
==============

Grad-CAM
    1. run the image forward, keep the feature maps A_k of the last encoder stage
    2. back-propagate the chosen output (e.g. the "Reject" logit) to get dScore/dA_k
    3. weight of each channel w_k = average of its gradient over the image
    4. CAM = ReLU( sum_k w_k * A_k ), upsampled and scaled to 0..1
    -> bright regions = pixels that pushed the model towards that output.

MC Dropout (Monte-Carlo dropout)
    Keep dropout ON at test time and run the heads N times. Each pass gives a
    slightly different answer. The spread tells us how unsure the model is:
      * mean probability of the top grade -> confidence
      * std of that probability          -> instability
      * predictive entropy               -> overall uncertainty (0 = certain)
    Low confidence / high std  =>  flag "needs manual review".
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer or model.cam_layer
        self._act = None
        self._grad = None
        self._h = self.layer.register_forward_hook(self._save)

    def _save(self, _m, _i, out):
        out = out[-1] if isinstance(out, (list, tuple)) else out
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def remove(self):
        self._h.remove()

    def __call__(self, x, target="grade", index=None):
        """x: 1x3xHxW. target: 'grade' (class `index`, default = predicted) or 'severity'.
        Returns (cam HxW float 0..1, model outputs)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            if target == "severity" and "severity" in out:
                score = out["severity"][0]
            else:
                logits = out["grade"][0]
                index = int(logits.argmax()) if index is None else index
                score = logits[index]
            self.model.zero_grad(set_to_none=True)
            score.backward()
        act, grad = self._act.detach()[0], self._grad.detach()[0]   # C x h x w
        w = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((w * act).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), {k: v.detach() for k, v in out.items()}


def _dropouts(model):
    return [m for m in model.modules() if isinstance(m, nn.Dropout)]


@torch.no_grad()
def mc_dropout(model, x, passes=20):
    """Run the model `passes` times with dropout ON (BatchNorm stays in eval mode).
    The encoder has no dropout, so it runs ONCE; only the small heads are repeated."""
    model.eval()
    feats = model.encoder(x)
    pooled = model.pool(feats[-1]).flatten(1)
    for d in _dropouts(model):
        d.train()
    probs, sevs = [], []
    try:
        for _ in range(passes):
            g = model.drop(pooled)
            probs.append(model.grade_head(g).softmax(1))
            if hasattr(model, "severity_head"):
                sevs.append(torch.sigmoid(model.severity_head(g)).squeeze(1))
    finally:
        model.eval()
    P = torch.stack(probs)                     # passes x B x n_grades
    mean = P.mean(0)
    top = mean.argmax(1)
    top_std = P[:, torch.arange(P.shape[1]), top].std(0)
    entropy = -(mean * (mean + 1e-12).log()).sum(1) / np.log(mean.shape[1])   # 0..1
    res = {"probs": mean, "top": top, "top_std": top_std, "entropy": entropy}
    if sevs:
        S = torch.stack(sevs)
        res["severity_mean"], res["severity_std"] = S.mean(0), S.std(0)
    seg = model.decoder(feats, x.shape[-2:]) if hasattr(model, "decoder") else None
    res["seg_logits"] = seg
    return res
