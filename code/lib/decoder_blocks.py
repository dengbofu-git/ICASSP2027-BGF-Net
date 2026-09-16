from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PVT_E1_CHANNELS = 64
PVT_ENC_CHANNELS = (128, 320, 512)


def weight_init(module):
    for _name, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(
            m,
            (
                nn.ReLU,
                nn.Sigmoid,
                nn.Softmax,
                nn.PReLU,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveMaxPool2d,
                nn.AdaptiveAvgPool1d,
                nn.Identity,
            ),
        ):
            pass
        elif hasattr(m, "initialize"):
            m.initialize()


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def initialize(self):
        weight_init(self)


class DecoderFusion(nn.Module):
    def __init__(self, input_count: int, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(input_count * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        target_size = features[0].shape[2:]
        aligned = [
            f if f.shape[2:] == target_size
            else F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in features
        ]
        return self.fuse(torch.cat(aligned, dim=1))

    def initialize(self):
        weight_init(self)


class BgSemanticHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 16)
        self.head = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, bg_feat: torch.Tensor) -> torch.Tensor:
        return self.head(bg_feat)

    def initialize(self):
        weight_init(self)
