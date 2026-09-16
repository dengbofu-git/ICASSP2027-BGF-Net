import numpy as np
import torch

METRIC_DECIMALS = 3


def round_metric(value):
    return round(float(value), METRIC_DECIMALS)


def format_metric(value):
    return f"{round_metric(value):.{METRIC_DECIMALS}f}"


def format_metrics(values):
    return [format_metric(v) for v in values]


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr_d(optimizer, decay_rate=0.5):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_rate


def format_optimizer_lrs(optimizer):
    parts = []
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{idx}")
        parts.append(f"{name}={group['lr']:.2e}")
    return ", ".join(parts)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu())
        else:
            val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        window = self.losses[np.maximum(len(self.losses) - self.num, 0):]
        if not window:
            return 0.0
        return float(np.mean(window))
