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


def test_bootstrap_ci():
    from fruitgrade.evaluate import comparison_table, headline_metrics
    from fruitgrade.metrics import ClsMeter, MAEMeter, bootstrap_ci
    rng = np.random.default_rng(0)
    cls, seg, mm, mh = ClsMeter(4), SegMeter(5), MAEMeter(), MAEMeter()
    for _ in range(60):                                   # one update() per image, like evaluate.py
        y = int(rng.integers(0, 4))
        cls.update([y], [y if rng.random() < 0.7 else (y + 1) % 4])
        gt = rng.integers(0, 5, (8, 8))
        seg.update(gt, np.where(rng.random((8, 8)) < 0.8, gt, rng.integers(0, 5, (8, 8))))
        mm.update([rng.uniform(0, 50)], [rng.uniform(0, 50)])
        mh.update([10.0], [12.0])
    def stat(idx):
        return headline_metrics("multitask", CFG, cls, seg, mm, mh, idx)
    full = stat(None)
    assert stat(np.arange(60)) == full                    # resampling every image once = point estimate
    ci = bootstrap_ci(60, stat, n_boot=300, seed=0)
    for k, v in full.items():
        lo, hi = ci[k]
        assert lo <= v <= hi, k
    assert ci["severity_mae_regression_head"] == [2.0, 2.0]   # constant error -> zero-width interval
    assert ci["accuracy"][1] - ci["accuracy"][0] > 0.05       # 60 noisy images -> visibly wide
    r = {"kind": "multitask", "classification": cls.result(CFG["grades"]),
         "segmentation": seg.result(CFG["seg_classes"]), "severity_mae_from_mask": full["severity_mae_from_mask"],
         "severity_mae_regression_head": 2.0, "review_flag_rate": 0.1, "errors_caught_by_flag": None, "ci95": ci}
    table = comparison_table([r])
    lo, hi = ci["accuracy"]
    assert f"[{lo * 100:.1f}, {hi * 100:.1f}]" in table and "2.00 [2.00, 2.00]" in table


def test_splits_have_no_group_leak(tmp_path, monkeypatch):
    import csv
    import sys
    from pathlib import Path

    import cv2
    import pytest
    from fruitgrade.data import prepare_kaggle
    from fruitgrade.data.dedupe import AUG_RE
    # 1. the tracked project split: originals only, no group in two splits
    rows = prepare_kaggle.load_splits(Path(__file__).parent.parent / "results/data_audit/splits.csv")
    assert len(rows) == 1511 and not any(AUG_RE.match(Path(r["path"]).name) for r in rows)
    prepare_kaggle.check_no_group_leak(rows)
    # 2. the check really fires
    with pytest.raises(ValueError):
        prepare_kaggle.check_no_group_leak([{"group": "7", "split": "train"}, {"group": "7", "split": "test"}])
    # 3. prepare_kaggle takes split + group from the splits file, never re-splits
    rng = np.random.default_rng(5)
    src, entries = tmp_path / "src", []
    for i, (folder, split, group) in enumerate([("freshapples", "train", "0"), ("rottenapples", "train", "1"),
                                                ("freshbanana", "val", "2"), ("rottenoranges", "test", "3")]):
        (src / "train" / folder).mkdir(parents=True, exist_ok=True)
        img = make_sample(160, 2, rng, RULES)[0]
        cv2.imwrite(str(src / "train" / folder / f"{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        entries.append({"path": f"train/{folder}/{i}.png", "split": split, "group": group})
    with open(tmp_path / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "group"])
        w.writeheader()
        w.writerows(entries)
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--splits", str(tmp_path / "splits.csv"),
                                      "--out", str(tmp_path / "out"), "--preview", "0"])
    prepare_kaggle.main()
    got = {r["src"]: (r["split"], r["group"]) for r in csv.DictReader(open(tmp_path / "out/labels.csv"))}
    skipped = [r["src"] for r in csv.DictReader(open(tmp_path / "out/skipped.csv"))]
    assert {e["path"] for e in entries} == set(got) | set(skipped) and len(got) >= 3
    for e in entries:
        if e["path"] in got:
            assert got[e["path"]] == (e["split"], e["group"])


def test_label_filter_flags_train_val_only(tmp_path):
    from fruitgrade.data.dataset import FruitDataset, read_labels
    from fruitgrade.data.label_filter import apply_filter
    from fruitgrade.data.synthetic import write_labels
    base = {"image": "images/x.jpg", "mask": "masks/x.png", "severity": 0, "fruit": "apple",
            "mask_quality": "pseudo", "group": "0", "src": "x"}
    cases = [("train", "kaggle_fresh", "Reject", "fresh_label_but_reject"),
             ("val", "kaggle_rotten", "A", "rotten_label_but_A"),
             ("train", "kaggle_fresh", "C", ""), ("val", "kaggle_rotten", "Reject", ""),
             ("test", "kaggle_fresh", "Reject", ""), ("test", "kaggle_rotten", "A", "")]
    rows = apply_filter([{**base, "split": s, "source": src, "grade": g} for s, src, g, _ in cases])
    assert [r["excluded_reason"] for r in rows] == [c[3] for c in cases]
    write_labels(tmp_path / "labels.csv", rows)
    assert len(read_labels(tmp_path)) == 4 and len(read_labels(tmp_path, include_excluded=True)) == 6
    assert len(FruitDataset(tmp_path, "train", CFG["grades"])) == 1      # the flagged train row is skipped
    assert len(FruitDataset(tmp_path, "test", CFG["grades"])) == 2       # test is never filtered


def test_grading_rules_locked():
    # Locked at commit 0e11134 (2026-10-19), before any real test evaluation.
    # Changing them needs the project owner's explicit approval (see CLAUDE.md); update both together.
    assert CFG["grading_rules"] == {"reject": {"total_pct": 30.0, "rot_pct": 10.0},
                                    "c": {"total_pct": 10.0, "rot_pct": 2.0},
                                    "b": {"total_pct": 2.0}}
    assert CFG["grades"] == ["A", "B", "C", "Reject"]
