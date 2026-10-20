"""Fast checks that every piece fits together. Run:  pytest -q"""
from collections import Counter

import numpy as np
import torch

from fruitgrade.config import load_config
from fruitgrade.data.pseudo_masks import pseudo_mask
from fruitgrade.data.synthetic import make_sample
from fruitgrade.explain import GradCAM, mc_dropout
from fruitgrade.grading import grade_from_severity, severity_from_mask
from fruitgrade.losses import MultiTaskLoss
from fruitgrade.metrics import SegMeter
from fruitgrade.models import BaselineClassifier, MultiTaskFruitNet

CFG = load_config()
RULES = CFG["grading_rules"]


def test_severity_and_rules():
    m = np.zeros((10, 10), np.uint8)
    m[:, :] = 1
    m[0, :5] = 3                          # 5 rot pixels out of 100 fruit pixels
    s = severity_from_mask(m)
    assert abs(s["total_pct"] - 5) < 1e-6 and abs(s["rot_pct"] - 5) < 1e-6
    assert CFG["grades"][grade_from_severity(s, RULES)] == "C"   # rot >= 2%
    m[0, :5] = 2                          # same area but bruise -> B
    assert CFG["grades"][grade_from_severity(severity_from_mask(m), RULES)] == "B"
    m[:] = 1
    assert CFG["grades"][grade_from_severity(severity_from_mask(m), RULES)] == "A"


def test_synthetic_sample_consistent():
    rng = np.random.default_rng(1)
    img, mask, sev, g, _ = make_sample(128, 3, rng, RULES)
    assert img.shape == (128, 128, 3) and mask.shape == (128, 128)
    assert g == grade_from_severity(severity_from_mask(mask), RULES)


def test_pseudo_mask_finds_fruit():
    rng = np.random.default_rng(2)
    img, mask, *_ = make_sample(160, 2, rng, RULES)
    pm = pseudo_mask(img, is_rotten=True)
    inter = ((pm > 0) & (mask > 0)).sum()
    union = ((pm > 0) | (mask > 0)).sum()
    assert inter / union > 0.85


def test_models_loss_cam_mc():
    x = torch.randn(2, 3, 128, 128)
    mt = MultiTaskFruitNet("mobilenetv3_large_100", pretrained=False)
    out = mt(x)
    assert out["grade"].shape == (2, 4) and out["seg"].shape == (2, 5, 128, 128)
    assert ((out["severity"] >= 0) & (out["severity"] <= 1)).all()
    batch = {"grade": torch.tensor([0, 3]), "mask": torch.randint(0, 5, (2, 128, 128)),
             "severity": torch.tensor([0.0, 0.4])}
    loss, parts = MultiTaskLoss()(out, batch)
    loss.backward()
    assert set(parts) == {"cls", "seg", "sev", "total"}

    cam, _ = GradCAM(mt)(x[:1])
    assert cam.shape == (128, 128) and 0 <= cam.min() and cam.max() <= 1
    mc = mc_dropout(mt, x[:1], passes=5)
    assert torch.allclose(mc["probs"].sum(1), torch.ones(1), atol=1e-5)

    bl = BaselineClassifier("mobilenetv3_large_100", pretrained=False)
    assert bl(x)["grade"].shape == (2, 4)
    cam, _ = GradCAM(bl)(x[:1])
    assert cam.shape == (128, 128)


def test_seg_metrics():
    gt = np.array([[0, 1, 2, 2]])
    pr = np.array([[0, 1, 2, 1]])
    m = SegMeter(5)
    m.update(gt, pr)
    r = m.result(CFG["seg_classes"])
    assert abs(r["iou"]["bruise"] - 0.5) < 1e-6
    assert abs(r["dice"]["bruise"] - 2 / 3) < 1e-6


import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gradcam_on_gpu():
    mt = MultiTaskFruitNet("mobilenetv3_large_100", pretrained=False).cuda().eval()
    cam, _ = GradCAM(mt)(torch.randn(1, 3, 128, 128, device="cuda"))
    assert cam.shape == (128, 128)


def test_invariant_phash_and_pairs():
    from pathlib import Path
    from PIL import Image
    from fruitgrade.data.dedupe import invariant_phash, near_pairs, original_key, split_groups
    rng = np.random.default_rng(3)
    a = Image.fromarray(make_sample(128, 2, rng, RULES)[0])
    ha = invariant_phash(a)
    assert ha == invariant_phash(a.transpose(Image.Transpose.ROTATE_90))
    assert ha == invariant_phash(a.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    # pairs are listed, never chained: 0~1 and 1~2 do not imply 0~2
    assert near_pairs([0b0, 0b111, 0b111111], max_dist=3) == [(0, 1, 3), (1, 2, 3)]
    assert original_key(Path("train/freshapples/rotated_by_15_X 1.png")) == ("freshapples/X 1.png", "rotated_by_15_")
    assert original_key(Path("test/rottenbanana/X.png")) == ("rottenbanana/X.png", "")
    # stratified group split: groups stay whole, ~15% test / ~15% val per class
    cls = {g: "a" if g < 100 else "b" for g in range(200)}
    s = split_groups(cls, {g: 1 for g in cls}, {"val": 0.15, "test": 0.15}, seed=42)
    for c in "ab":
        got = Counter(v for g, v in s.items() if cls[g] == c)
        assert got == {"train": 70, "val": 15, "test": 15}
