import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.baseline import format_learnable_weights
from lib.bgf_losses import (
    DEFAULT_MMD_SIGMAS,
    background_semantic_dice,
    compute_bgf_losses,
    finalize_domain_mmd_metrics,
    make_reliable_supervision_masks,
    pool_domain_mmd_features,
)
from utils.dataloader import DOMAIN_CVC_CLINIC, DOMAIN_KVASIR
from utils.utils import (
    adjust_lr_d,
    clip_gradient,
    AvgMeter,
    format_metric,
    format_optimizer_lrs,
    round_metric,
)

torch.backends.cudnn.enabled = False


def dice_score(pred, mask, smooth=1.0):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask,
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def build_optimizer(model, lr, optimizer_name="AdamW"):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=1e-4, momentum=0.9,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for group in optimizer.param_groups:
        group["base_lr"] = lr
        group["name"] = "all"
    return optimizer


def bg_aux_scale(epoch, warmup_epochs):
    warm = max(int(warmup_epochs), 1)
    return min(1.0, float(epoch) / float(warm))


def allocation_temperature(epoch, tau_epochs=100):
    t = max(int(tau_epochs), 1)
    return max(0.7, 1.5 - 0.8 * float(epoch) / float(t))


def add_training_args(parser):
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD"],
    )
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_min_epochs", type=int, default=60)
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--train_path", type=str, default="./data/TrainDataset")
    parser.add_argument("--test_path", type=str, default="./data/TestDataset")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--lambda_bg_sem", type=float, default=0.2)
    parser.add_argument("--lambda_bg_mask", type=float, default=0.2)
    parser.add_argument("--lambda_bg_inv", type=float, default=0.05)
    parser.add_argument("--lambda_bg_dice", type=float, default=1.0)
    parser.add_argument("--aux_seg_weight", type=float, default=0.2)
    parser.add_argument("--mmd_sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--bg_radius", type=int, default=5)
    parser.add_argument("--bg_aux_warmup", type=int, default=20)
    parser.add_argument("--allocation_tau_epochs", type=int, default=100)
    parser.add_argument("--fg_keep_eps", type=float, default=0.05)


def _resolve_mmd_sigmas(opt) -> tuple[float, ...]:
    if getattr(opt, "mmd_sigmas", None):
        return tuple(float(s) for s in opt.mmd_sigmas)
    return DEFAULT_MMD_SIGMAS


def validate(val_loader, model, device, opt=None):
    model.eval()
    scores_k = []
    scores_c = []
    bg_radius = getattr(opt, "bg_radius", 5) if opt is not None else 5
    mmd_sigmas = _resolve_mmd_sigmas(opt) if opt is not None else DEFAULT_MMD_SIGMAS
    bg_pooled = {DOMAIN_KVASIR: [], DOMAIN_CVC_CLINIC: []}
    fg_pooled = {DOMAIN_KVASIR: [], DOMAIN_CVC_CLINIC: []}
    bg_sem_dice_vals = []

    with torch.no_grad():
        for batch in val_loader:
            images, gts = batch[0], batch[1]
            domain_id = int(batch[2].item()) if len(batch) > 2 else None
            images = images.to(device)
            gts = gts.to(device)
            outputs, bg_pack = model(images, return_bg=True, temperature=1.0)
            p_out = outputs["lout"]
            dice = round_metric(float(dice_score(p_out, gts).cpu()))
            if domain_id == DOMAIN_KVASIR:
                scores_k.append(dice)
            elif domain_id == DOMAIN_CVC_CLINIC:
                scores_c.append(dice)

            if domain_id is not None and "bg_feat" in bg_pack:
                m_bg, m_fg, _m_valid = make_reliable_supervision_masks(gts, radius=bg_radius)
                fg_list = bg_pack.get("fg_feat") or bg_pack["bg_feat"]
                z_bg, z_fg = pool_domain_mmd_features(
                    bg_pack["bg_feat"], fg_list, m_bg, m_fg,
                )
                bg_pooled[domain_id].append(z_bg.cpu())
                fg_pooled[domain_id].append(z_fg.cpu())
                bg_sem_dice_vals.append(
                    background_semantic_dice(bg_pack["bg_logit"], gts, radius=bg_radius)
                )

    model.train()

    d_k = round_metric(sum(scores_k) / max(len(scores_k), 1))
    d_c = round_metric(sum(scores_c) / max(len(scores_c), 1))
    s_id = round_metric(0.5 * d_k + 0.5 * d_c)
    stats = {
        "d_k": d_k,
        "d_c": d_c,
        "s_id": s_id,
        "n_k": len(scores_k),
        "n_c": len(scores_c),
    }
    if bg_sem_dice_vals:
        stats["bg_sem_dice"] = round_metric(
            sum(bg_sem_dice_vals) / len(bg_sem_dice_vals)
        )
    mmd_stats = finalize_domain_mmd_metrics(bg_pooled, fg_pooled, mmd_sigmas=mmd_sigmas)
    stats.update({k: round_metric(v) for k, v in mmd_stats.items()})
    return s_id, stats


def run_training_loop(model, optimizer, train_loader, val_loader, opt, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    total_epochs = opt.epoch
    save_start_epoch = total_epochs // 4
    min_epochs = getattr(opt, "early_stop_min_epochs", 60)
    patience = getattr(opt, "early_stop_patience", 50)
    save_delta = getattr(opt, "save_delta", 1e-4)
    decay_rate = getattr(opt, "decay_rate", 0.5)

    best_val_sid = float("-inf")
    best_epoch = 0
    n_bad = 0
    stopped_early = False

    print(
        "optimizer: {}, lr={:.1e}, lr decay x{} at epochs [50, 90, 120]; "
        "checkpoint: S_ID=(D_K+D_C)/2, start after epoch {}, delta={:.1e}; "
        "early stop: min_epochs={}, patience={}; "
        "bg_aux: warmup={} epoch".format(
            getattr(opt, "optimizer", "AdamW"), opt.lr, decay_rate,
            save_start_epoch, save_delta, min_epochs, patience,
            getattr(opt, "bg_aux_warmup", 20),
        )
    )

    for epoch in tqdm(range(1, opt.epoch), desc="Training"):
        if epoch in (50, 90, 120):
            adjust_lr_d(optimizer, decay_rate)
        train_loss = train(train_loader, model, optimizer, epoch, opt)
        val_sid, val_stats = validate(val_loader, model, opt.device, opt=opt)
        mmd_msg = ""
        if "mmd_bg" in val_stats:
            mmd_msg = ", MMD_bg={}".format(format_metric(val_stats["mmd_bg"]))
        if "mmd_fg" in val_stats:
            mmd_msg += ", MMD_fg={}".format(format_metric(val_stats["mmd_fg"]))
        if "bg_sem_dice" in val_stats:
            mmd_msg += ", Dice_bg={}".format(format_metric(val_stats["bg_sem_dice"]))
        if "var_bg" in val_stats:
            mmd_msg += ", Var_bg={}".format(format_metric(val_stats["var_bg"]))
        print(
            "Epoch {}/{}, train_loss={:.6f}, "
            "D_K={}, D_C={}, S_ID={}{}, lr=[{}], n_bad={}".format(
                epoch, total_epochs - 1, train_loss,
                format_metric(val_stats["d_k"]),
                format_metric(val_stats["d_c"]),
                format_metric(val_sid),
                mmd_msg,
                format_optimizer_lrs(optimizer), n_bad,
            )
        )

        if epoch > save_start_epoch:
            if val_sid > best_val_sid + save_delta:
                best_val_sid = val_sid
                best_epoch = epoch
                save_path = os.path.join(save_dir, "best.pth")
                torch.save(model.state_dict(), save_path)
                print(
                    "[保存最优模型] {} (epoch {}, S_ID={}, D_K={}, D_C={})".format(
                        save_path, epoch,
                        format_metric(val_sid),
                        format_metric(val_stats["d_k"]),
                        format_metric(val_stats["d_c"]),
                    )
                )
                n_bad = 0
            else:
                n_bad += 1

        if epoch >= min_epochs and n_bad >= patience:
            print(
                "Early stopping at epoch {} "
                "(S_ID improvement < {:.1e} for {} epochs)".format(
                    epoch, save_delta, patience,
                )
            )
            stopped_early = True
            break

    if best_epoch == 0:
        raise FileNotFoundError(
            "训练结束但未保存 best.pth；"
            "请检查 epoch 是否大于 {} 且 S_ID 有改进".format(save_start_epoch)
        )

    status = "early stopped" if stopped_early else "completed"
    print(
        "训练{}，最优 epoch={}，S_ID={}".format(
            status, best_epoch, format_metric(best_val_sid),
        )
    )
    return os.path.join(save_dir, "best.pth"), best_epoch, best_val_sid


def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    device = getattr(opt, "device", "cuda" if torch.cuda.is_available() else "cpu")
    sampler = getattr(train_loader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    size_rates = [0.75, 1, 1.25]
    meters = {k: AvgMeter() for k in (
        "p2", "p3", "p4", "lout", "l_mask", "l_bg_sem", "l_inv", "total",
    )}
    print("current learning rate:", format_optimizer_lrs(optimizer))
    total_step = len(train_loader)
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    aux_scale = bg_aux_scale(epoch, getattr(opt, "bg_aux_warmup", 20))
    lambda_sem = getattr(opt, "lambda_bg_sem", 0.2)
    lambda_mask = getattr(opt, "lambda_bg_mask", 0.2)
    lambda_inv = getattr(opt, "lambda_bg_inv", 0.05)
    lambda_dice = getattr(opt, "lambda_bg_dice", 1.0)
    aux_seg_w = getattr(opt, "aux_seg_weight", 0.2)
    mmd_sigmas = _resolve_mmd_sigmas(opt)
    tau_epochs = getattr(opt, "allocation_tau_epochs", 100)
    temperature = allocation_temperature(epoch, tau_epochs)

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts, domain_ids = pack
            images = images.to(device)
            gts = gts.to(device)
            domain_ids = domain_ids.to(device)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images, size=(trainsize, trainsize),
                    mode="bilinear", align_corners=True,
                )
                gts = F.interpolate(
                    gts, size=(trainsize, trainsize),
                    mode="bilinear", align_corners=True,
                )

            outputs, bg_pack = model(
                images, return_bg=True, temperature=temperature,
            )
            p2 = outputs["p2"]
            p3 = outputs["p3"]
            p4 = outputs["p4"]
            lout = outputs["lout"]

            l2 = structure_loss(p2, gts)
            l3 = structure_loss(p3, gts)
            l4 = structure_loss(p4, gts)
            l_main = structure_loss(lout, gts)
            l_seg = l_main + aux_seg_w * (l2 + l3 + l4)

            bg_losses = compute_bgf_losses(
                bg_pack["bg_feat"],
                bg_pack["bg_logit"],
                gt_mask=gts,
                domain_ids=domain_ids,
                m_fg_list=bg_pack["m_fg"],
                m_bg_list=bg_pack["m_bg"],
                bg_radius=getattr(opt, "bg_radius", 5),
                mmd_sigmas=mmd_sigmas,
                lambda_dice=lambda_dice,
            )
            l_bg_sem = bg_losses["bg_sem"]
            l_inv = bg_losses["inv"]
            l_mask = bg_losses["mask"]
            loss = (
                l_seg
                + lambda_mask * l_mask
                + lambda_sem * l_bg_sem
                + aux_scale * lambda_inv * l_inv
            )
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                batch_loss = float(loss.detach().cpu())
                epoch_loss_sum += batch_loss
                epoch_batch_count += 1
                meters["p2"].update(l2.data, opt.batchsize)
                meters["p3"].update(l3.data, opt.batchsize)
                meters["p4"].update(l4.data, opt.batchsize)
                meters["lout"].update(l_main.data, opt.batchsize)
                meters["l_mask"].update(l_mask.data, opt.batchsize)
                meters["l_bg_sem"].update(l_bg_sem.data, opt.batchsize)
                meters["l_inv"].update(l_inv.data, opt.batchsize)
                meters["total"].update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], "
                "[p2: {:.4f}, p3: {:.4f}, p4: {:.4f}, "
                "lout: {:.4f}, L_mask: {:.4f}, L_bg-sem: {:.4f}, L_inv: {:.4f}] "
                "aux_scale={:.2f} tau={:.2f}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step,
                    meters["p2"].show(), meters["p3"].show(),
                    meters["p4"].show(), meters["lout"].show(),
                    meters["l_mask"].show(), meters["l_bg_sem"].show(),
                    meters["l_inv"].show(), aux_scale, temperature,
                )
            )

    epoch_avg_loss = epoch_loss_sum / max(epoch_batch_count, 1)
    print(
        "Cur Epoch [{:03d}/{:03d}], epoch_avg_loss: {:.6f}, step_avg_loss: {:.4f}, {}".format(
            epoch, opt.epoch, epoch_avg_loss, meters["total"].show(),
            format_learnable_weights(model),
        )
    )
    return epoch_avg_loss
