"""
Train the multi-task model or the classification-only baseline.

    python -m fruitgrade.train --model multitask
    python -m fruitgrade.train --model baseline
    # quick CPU smoke test:
    python -m fruitgrade.train --model multitask --epochs 2 --no-pretrained --limit 64

Saves to runs/<model>_<backbone>/ :
    best.pt        best checkpoint (by validation score), config stored inside
    history.csv    per-epoch losses and metrics
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fruitgrade.config import load_config
from fruitgrade.data.dataset import FruitDataset
from fruitgrade.grading import severity_from_mask
from fruitgrade.losses import MultiTaskLoss
from fruitgrade.metrics import ClsMeter, MAEMeter, SegMeter
from fruitgrade.models import build_model


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_eval(model, loader, cfg, device):
    """Validation pass -> metrics dict (same metrics as evaluate.py)."""
    model.eval()
    cls = ClsMeter(len(cfg["grades"]))
    seg = SegMeter(len(cfg["seg_classes"]))
    mae_head, mae_mask = MAEMeter(), MAEMeter()
    for b in loader:
        out = model(b["image"].to(device))
        cls.update(b["grade"].numpy(), out["grade"].argmax(1).cpu().numpy())
        if "seg" in out:
            pm = out["seg"].argmax(1).cpu().numpy()
            seg.update(b["mask"].numpy(), pm)
            true_pct = b["severity"].numpy() * 100
            mae_head.update(true_pct, out["severity"].cpu().numpy() * 100)
            mae_mask.update(true_pct, [severity_from_mask(m)["total_pct"] for m in pm])
    res = {"cls": cls.result(cfg["grades"])}
    if seg.tp.sum() > 0:
        res["seg"] = seg.result(cfg["seg_classes"])
        res["severity_mae_head"] = mae_head.result()
        res["severity_mae_mask"] = mae_mask.result()
    return res


def score(res):
    """One number to pick the best epoch: grade F1 (+ defect IoU, - severity error)."""
    s = res["cls"]["macro_f1"]
    if "seg" in res:
        s += res["seg"]["mean_defect_iou"] - res["severity_mae_head"] / 100
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["multitask", "baseline"], default="multitask")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only N train images (debug)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]
    if args.epochs:
        tc["epochs"] = args.epochs
    if args.batch_size:
        tc["batch_size"] = args.batch_size
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    set_seed(tc["seed"])
    device = get_device(args.device)

    root, size = cfg["data"]["root"], cfg["data"]["image_size"]
    train_ds = FruitDataset(root, "train", cfg["grades"], size, train=True)
    val_ds = FruitDataset(root, "val", cfg["grades"], size, train=False)
    counts = train_ds.grade_counts()
    if args.limit:
        train_ds = Subset(train_ds, list(range(min(args.limit, len(train_ds)))))
    nw = cfg["data"]["num_workers"]
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, num_workers=nw, drop_last=True)
    val_dl = DataLoader(val_ds, tc["batch_size"], shuffle=False, num_workers=nw)

    model = build_model(args.model, cfg).to(device)
    # inverse-frequency class weights for the grade loss
    w = counts.sum() / np.maximum(counts, 1) / len(counts)
    grade_w = torch.tensor(w, dtype=torch.float32, device=device)
    seg_w = torch.tensor([0.5, 0.5, 2.0, 2.0, 2.0], device=device)  # emphasise defect pixels
    loss_fn = MultiTaskLoss(tc["w_cls"], tc["w_seg"], tc["w_sev"], grade_w, seg_w)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    # pretrained backbone -> learn slowly; random-init backbone -> same LR as heads
    enc_mult = tc["backbone_lr_mult"] if cfg["model"]["pretrained"] else 1.0
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": tc["lr"] * enc_mult},
        {"params": head_params, "lr": tc["lr"]},
    ], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])

    out_dir = Path(args.out or Path(tc["out_dir"]) / f"{args.model}_{cfg['model']['backbone']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | train {len(train_ds)} | val {len(val_ds)} | grade counts {counts.tolist()}")
    print(f"Saving to {out_dir}")

    best, bad_epochs, hist = -1e9, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        freeze = epoch <= tc["freeze_backbone_epochs"] and cfg["model"]["pretrained"]
        for p in enc_params:
            p.requires_grad = not freeze
        model.train()
        t0, sums, n = time.time(), {}, 0
        for b in train_dl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["image"])
            loss, parts = loss_fn(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0) + v
            n += 1
        sched.step()
        tr = {f"train_{k}": v / max(n, 1) for k, v in sums.items()}
        res = run_eval(model, val_dl, cfg, device)
        sc = score(res)
        row = {"epoch": epoch, **tr, "val_acc": res["cls"]["accuracy"], "val_f1": res["cls"]["macro_f1"],
               "val_defect_iou": res.get("seg", {}).get("mean_defect_iou", ""),
               "val_defect_dice": res.get("seg", {}).get("mean_defect_dice", ""),
               "val_sev_mae": res.get("severity_mae_head", ""), "score": sc,
               "sec": round(time.time() - t0, 1), "frozen": freeze}
        hist.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

        if sc > best:
            best, bad_epochs = sc, 0
            torch.save({"kind": args.model, "config": cfg, "state_dict": model.state_dict(),
                        "epoch": epoch, "val": res}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc["early_stop_patience"]:
                print("Early stopping.")
                break

    with open(out_dir / "history.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        wr.writeheader()
        wr.writerows(hist)
    json.dump({"best_score": best}, open(out_dir / "summary.json", "w"), indent=2)
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
