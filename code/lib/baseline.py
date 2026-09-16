import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.bgf_decoder import BGFDecoder
from lib.pvtv2 import pvt_v2_b2

FIXED_PVT_MODEL = "b2"
PVT_ENC_CHANNELS = (128, 320, 512)


def format_learnable_weights(model):
    return (
        f"backbone=PVTv2-{FIXED_PVT_MODEL.upper()}, "
        f"decoder=BGF-Net, "
        f"fg_keep_eps={getattr(model, 'fg_keep_eps', 0.05):.4f}"
    )


def build_backbone(pretrained=True):
    backbone = pvt_v2_b2()
    if not pretrained:
        return backbone

    path = f"./models/pvt_v2_{FIXED_PVT_MODEL}.pth"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 PVTv2-{FIXED_PVT_MODEL.upper()} 预训练权重: {path}"
        )

    save_model = torch.load(path, map_location="cpu")
    model_dict = backbone.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict}
    if not state_dict:
        raise RuntimeError(
            f"预训练权重 {path} 与 PVTv2-{FIXED_PVT_MODEL.upper()} 结构不匹配"
        )
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)
    return backbone


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class AddCoords(nn.Module):
    def forward(self, x):
        batch, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        yy = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy = yy.view(1, 1, h, 1).expand(batch, 1, h, w)
        xx = xx.view(1, 1, 1, w).expand(batch, 1, h, w)
        return torch.cat([x, yy, xx], dim=1)


class BasicCoordConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.add_coords = AddCoords()
        self.conv = nn.Conv2d(
            in_planes + 2, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class CoordConv_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicCoordConv2d(out_channel, out_channel, 3, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicCoordConv2d(out_channel, out_channel, 3, padding=2, dilation=2),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicCoordConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x_cat = self.conv_cat(torch.cat([
            self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x),
        ], dim=1))
        return self.relu(x_cat + self.conv_res(x))


class PartialDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

    def forward(self, x4_coord, x3_coord, x2_coord):
        x1_1 = x4_coord
        x2_1 = self.conv_upsample1(self.upsample(x4_coord)) * x3_coord
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x4_coord)))
            * self.conv_upsample3(self.upsample(x3_coord))
            * x2_coord
        )
        x2_2 = self.conv_concat2(torch.cat([
            x2_1, self.conv_upsample4(self.upsample(x1_1)),
        ], dim=1))
        x3_2 = self.conv_concat3(torch.cat([
            x3_1, self.conv_upsample5(self.upsample(x2_2)),
        ], dim=1))
        return self.conv4(x3_2)


class BaseModel(nn.Module):
    def __init__(
        self,
        channel=32,
        num_class=1,
        pretrained=True,
        dec_channels=128,
        fg_keep_eps=0.05,
    ):
        super().__init__()
        self.num_class = num_class
        self.model = FIXED_PVT_MODEL
        self.fg_keep_eps = float(fg_keep_eps)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.backbone = build_backbone(pretrained=pretrained)

        e2_ch, e3_ch, e4_ch = PVT_ENC_CHANNELS
        self.coord2 = CoordConv_modified(e2_ch, channel)
        self.coord3 = CoordConv_modified(e3_ch, channel)
        self.coord4 = CoordConv_modified(e4_ch, channel)
        self.pd = PartialDecoder(channel)
        e5_ch = 3 * channel
        self.decoder = BGFDecoder(
            channels=dec_channels,
            e5_ch=e5_ch,
            num_class=num_class,
            fg_keep_eps=fg_keep_eps,
        )

    def forward(
        self,
        x,
        return_feats=False,
        return_bg=False,
        temperature: float = 1.0,
    ):
        if x.size(1) == 1:
            x = self.conv(x)

        e1, e2, e3, e4 = self.backbone(x)
        x2_coord = self.coord2(e2)
        x3_coord = self.coord3(e3)
        x4_coord = self.coord4(e4)

        e5_pd = self.pd(x4_coord, x3_coord, x2_coord)
        dec_out = self.decoder(
            e1=e1,
            e2=e2,
            e3=e3,
            e4=e4,
            e5_pd=e5_pd,
            shape=x.shape[2:],
            temperature=temperature,
        )

        outputs = {
            "p2": dec_out["p2"],
            "p3": dec_out["p3"],
            "p4": dec_out["p4"],
            "lout": dec_out["lout"],
        }

        bg_pack = {
            "bg_feat": dec_out["bg_feat"],
            "fg_feat": dec_out["fg_feat"],
            "fg_suppressed": dec_out["fg_suppressed"],
            "bg_logit": dec_out["bg_logit"],
            "m_fg": dec_out["m_fg"],
            "m_bg": dec_out["m_bg"],
        }

        if return_feats:
            d4, d3, d2, d1 = dec_out["decoder_stages"]
            f4, f3, f2 = dec_out["fg_feat"]
            f4_t, f3_t, f2_t = dec_out["fg_suppressed"]
            bg_logit4, bg_logit3, bg_logit2 = dec_out["bg_logit"]
            m4_fg, m3_fg, m2_fg = dec_out["m_fg"]
            m4_bg, m3_bg, m2_bg = dec_out["m_bg"]
            feats = {
                "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5_pd": e5_pd,
                "d4": d4, "d3": d3, "d2": d2, "d1": d1,
                "bg4": dec_out["bg_feat"][0],
                "bg3": dec_out["bg_feat"][1],
                "bg2": dec_out["bg_feat"][2],
                "fg4": f4,
                "fg3": f3,
                "fg2": f2,
                "fg4_suppressed": f4_t,
                "fg3_suppressed": f3_t,
                "fg2_suppressed": f2_t,
                "bg_logit4": bg_logit4,
                "bg_logit3": bg_logit3,
                "bg_logit2": bg_logit2,
                "m_fg4": m4_fg,
                "m_fg3": m3_fg,
                "m_fg2": m2_fg,
                "m_bg4": m4_bg,
                "m_bg3": m3_bg,
                "m_bg2": m2_bg,
            }
            if return_bg:
                return outputs, bg_pack, feats
            return outputs, feats

        if return_bg:
            return outputs, bg_pack
        return outputs
