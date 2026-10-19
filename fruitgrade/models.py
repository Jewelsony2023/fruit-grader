"""
Models
======

MultiTaskFruitNet  (proposed)
    image -> shared pretrained CNN encoder (timm: ResNet-50 / EfficientNet-B0 / MobileNetV3)
           |-> grade head      : global pool -> dropout -> FC -> 4 logits (A/B/C/Reject)
           |-> severity head   : global pool -> dropout -> FC -> sigmoid -> 0..1 (x100 = %)
           '-> seg decoder     : U-Net style decoder with skip connections -> 5-class mask

BaselineClassifier (comparison)
    same encoder + grade head only. No mask, no severity head.

Both expose `.cam_layer` (last encoder stage) for Grad-CAM.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_encoder(name, pretrained):
    enc = timm.create_model(name, pretrained=pretrained, features_only=True)
    return enc, enc.feature_info.channels()


@torch.no_grad()
def find_cam_layer(encoder):
    """Find the module that produces the deepest feature map (used by Grad-CAM).
    Works for any timm backbone: run a dummy image, record every module's output,
    keep the LAST module whose output equals the final feature map's shape."""
    was_training = encoder.training
    encoder.eval()
    calls, hooks = [], []
    for name, m in encoder.named_modules():
        if name and name.count(".") <= 1:
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, n=name: calls.append((n, mod, o.shape if torch.is_tensor(o) else None))))
    device = next(encoder.parameters()).device
    feats = encoder(torch.zeros(1, 3, 128, 128, device=device))
    for h in hooks:
        h.remove()
    encoder.train(was_training)
    target = feats[-1].shape
    for name, mod, shape in reversed(calls):
        if shape == target:
            return mod
    raise RuntimeError("Could not find a Grad-CAM layer")


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNetDecoder(nn.Module):
    """Upsample the deepest feature map step by step, adding skip connections."""

    def __init__(self, enc_channels, dec_channels, n_classes):
        super().__init__()
        enc_channels = list(enc_channels)[::-1]          # deepest first
        skips = enc_channels[1:] + [0]
        ins = [enc_channels[0]] + list(dec_channels[:-1])
        n = min(len(dec_channels), len(enc_channels))
        self.blocks = nn.ModuleList(
            ConvBlock(ins[i] + skips[i], dec_channels[i]) for i in range(n))
        self.head = nn.Conv2d(dec_channels[n - 1], n_classes, 1)

    def forward(self, feats, out_size):
        feats = feats[::-1]
        x = feats[0]
        for i, block in enumerate(self.blocks):
            skip = feats[i + 1] if i + 1 < len(feats) else None
            if skip is not None:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], 1)
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        return F.interpolate(self.head(x), size=out_size, mode="bilinear", align_corners=False)


class MultiTaskFruitNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, n_grades=4, n_seg=5,
                 dropout=0.3, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder, ch = make_encoder(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.grade_head = nn.Sequential(nn.Linear(ch[-1], 256), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(256, n_grades))
        self.severity_head = nn.Sequential(nn.Linear(ch[-1], 128), nn.ReLU(inplace=True),
                                           nn.Dropout(dropout), nn.Linear(128, 1))
        self.decoder = UNetDecoder(ch, decoder_channels, n_seg)

    @property
    def cam_layer(self):
        return find_cam_layer(self.encoder)

    def forward(self, x):
        feats = self.encoder(x)
        g = self.drop(self.pool(feats[-1]).flatten(1))
        return {
            "grade": self.grade_head(g),
            "severity": torch.sigmoid(self.severity_head(g)).squeeze(1),   # 0..1
            "seg": self.decoder(feats, x.shape[-2:]),
        }


class BaselineClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, n_grades=4, dropout=0.3, **_):
        super().__init__()
        self.encoder, ch = make_encoder(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.grade_head = nn.Sequential(nn.Linear(ch[-1], 256), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(256, n_grades))

    @property
    def cam_layer(self):
        return find_cam_layer(self.encoder)

    def forward(self, x):
        feats = self.encoder(x)
        return {"grade": self.grade_head(self.drop(self.pool(feats[-1]).flatten(1)))}


def build_model(kind, cfg, pretrained=None):
    m = cfg["model"]
    kw = dict(backbone=m["backbone"],
              pretrained=m["pretrained"] if pretrained is None else pretrained,
              n_grades=len(cfg["grades"]), dropout=m["dropout"])
    if kind == "multitask":
        return MultiTaskFruitNet(n_seg=len(cfg["seg_classes"]),
                                 decoder_channels=tuple(m["decoder_channels"]), **kw)
    if kind == "baseline":
        return BaselineClassifier(**kw)
    raise ValueError(kind)


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint saved by train.py (config is stored inside)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["kind"], ck["config"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
