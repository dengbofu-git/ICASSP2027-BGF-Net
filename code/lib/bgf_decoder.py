from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoder_blocks import (
    BgSemanticHead,
    ConvBNReLU,
    DecoderFusion,
    PVT_E1_CHANNELS,
    PVT_ENC_CHANNELS,
    weight_init,
)


class BackgroundGuidedAllocation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alloc = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )
        self.fg_proj = ConvBNReLU(channels, channels)
        self.bg_proj = ConvBNReLU(channels, channels)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.shape[2:] != z.shape[2:]:
            context = F.interpolate(
                context, size=z.shape[2:], mode="bilinear", align_corners=False,
            )
        logits = self.alloc(torch.cat([z, context], dim=1))
        masks = F.softmax(logits / max(float(temperature), 1e-3), dim=1)
        m_fg = masks[:, 0:1]
        m_bg = masks[:, 1:2]
        f = self.fg_proj(z) * m_fg
        b = self.bg_proj(z) * m_bg
        return f, b, m_fg, m_bg


class BGFDecoder(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        e5_ch: int = 96,
        num_class: int = 1,
        fg_keep_eps: float = 0.05,
    ):
        super().__init__()
        e2_ch, e3_ch, e4_ch = PVT_ENC_CHANNELS
        e1_ch = PVT_E1_CHANNELS

        self.channels = channels
        self.fg_keep_eps = float(fg_keep_eps)

        self.proj1 = ConvBNReLU(e1_ch, channels)
        self.proj2 = ConvBNReLU(e2_ch, channels)
        self.proj3 = ConvBNReLU(e3_ch, channels)
        self.proj4 = ConvBNReLU(e4_ch, channels)
        self.e5_proj = ConvBNReLU(e5_ch, channels)

        self.alloc4 = BackgroundGuidedAllocation(channels)
        self.alloc3 = BackgroundGuidedAllocation(channels)
        self.alloc2 = BackgroundGuidedAllocation(channels)

        self.bg_head4 = BgSemanticHead(channels)
        self.bg_head3 = BgSemanticHead(channels)
        self.bg_head2 = BgSemanticHead(channels)

        self.fuse4 = DecoderFusion(input_count=2, channels=channels)
        self.fuse3 = DecoderFusion(input_count=3, channels=channels)
        self.fuse2 = DecoderFusion(input_count=3, channels=channels)
        self.fuse1 = DecoderFusion(input_count=2, channels=channels)

        self.head_p4 = nn.Conv2d(channels, num_class, 3, padding=1)
        self.head_p3 = nn.Conv2d(channels, num_class, 3, padding=1)
        self.head_p2 = nn.Conv2d(channels, num_class, 3, padding=1)
        self.head_out = nn.Conv2d(channels, num_class, 3, padding=1)

        self.initialize()

    def forward(
        self,
        *,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
        e5_pd: torch.Tensor,
        shape: Tuple[int, int],
        temperature: float = 1.0,
    ) -> Dict[str, object]:
        z1 = self.proj1(e1)
        z2 = self.proj2(e2)
        z3 = self.proj3(e3)
        z4 = self.proj4(e4)

        c = self.e5_proj(e5_pd)
        c4 = F.interpolate(c, size=z4.shape[2:], mode="bilinear", align_corners=False)
        c3 = F.interpolate(c, size=z3.shape[2:], mode="bilinear", align_corners=False)
        c2 = c

        f4, b4, m4_fg, m4_bg = self.alloc4(z4, c4, temperature=temperature)
        f3, b3, m3_fg, m3_bg = self.alloc3(z3, c3, temperature=temperature)
        f2, b2, m2_fg, m2_bg = self.alloc2(z2, c2, temperature=temperature)

        eps = self.fg_keep_eps
        bg_logit4 = self.bg_head4(b4)
        bg_logit3 = self.bg_head3(b3)
        bg_logit2 = self.bg_head2(b2)

        def _suppress(f_s, bg_logit):
            q_bg = torch.sigmoid(bg_logit).detach()
            fg_weight = eps + (1.0 - eps) * (1.0 - q_bg)
            return f_s * fg_weight

        f4_suppressed = _suppress(f4, bg_logit4)
        f3_suppressed = _suppress(f3, bg_logit3)
        f2_suppressed = _suppress(f2, bg_logit2)

        d4 = self.fuse4(z4, f4_suppressed)
        d3 = self.fuse3(
            F.interpolate(d4, size=z3.shape[2:], mode="bilinear", align_corners=False),
            z3,
            f3_suppressed,
        )
        d2 = self.fuse2(
            F.interpolate(d3, size=z2.shape[2:], mode="bilinear", align_corners=False),
            z2,
            f2_suppressed,
        )
        d1 = self.fuse1(
            F.interpolate(d2, size=z1.shape[2:], mode="bilinear", align_corners=False),
            z1,
        )

        p4 = F.interpolate(self.head_p4(d4), size=shape, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.head_p3(d3), size=shape, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.head_p2(d2), size=shape, mode="bilinear", align_corners=False)
        lout = F.interpolate(self.head_out(d1), size=shape, mode="bilinear", align_corners=False)

        return {
            "p4": p4,
            "p3": p3,
            "p2": p2,
            "lout": lout,
            "bg_feat": [b4, b3, b2],
            "fg_feat": [f4, f3, f2],
            "fg_suppressed": [f4_suppressed, f3_suppressed, f2_suppressed],
            "bg_logit": [bg_logit4, bg_logit3, bg_logit2],
            "m_fg": [m4_fg, m3_fg, m2_fg],
            "m_bg": [m4_bg, m3_bg, m2_bg],
            "decoder_stages": [d4, d3, d2, d1],
        }

    def initialize(self):
        weight_init(self)
