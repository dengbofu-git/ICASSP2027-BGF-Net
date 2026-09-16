from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MMD_SIGMAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
DEFAULT_MMD_LAYER_INDEX = 1
MMD_INV_LAYER_INDICES: tuple[int, ...] = (0, 1)


def _to_fg_mask(gt_mask: torch.Tensor) -> torch.Tensor:
    if gt_mask.dim() == 3:
        gt_fg = gt_mask.unsqueeze(1)
    else:
        gt_fg = gt_mask.float()
    return (gt_fg > 0.5).float()


def _morph_dilate(fg: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * int(radius) + 1
    pad = int(radius)
    return F.max_pool2d(fg, kernel_size=k, stride=1, padding=pad)


def _morph_erode(fg: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * int(radius) + 1
    pad = int(radius)
    return 1.0 - F.max_pool2d(1.0 - fg, kernel_size=k, stride=1, padding=pad)


def make_reliable_supervision_masks(
    gt_mask: torch.Tensor,
    radius: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fg = _to_fg_mask(gt_mask)
    m_bg = 1.0 - _morph_dilate(fg, radius)
    m_fg = _morph_erode(fg, radius)
    m_valid = (m_bg + m_fg).clamp(0.0, 1.0)
    return m_bg, m_fg, m_valid


def align_mask_nearest(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.shape[2:] != ref.shape[2:]:
        mask = F.interpolate(mask, size=ref.shape[2:], mode="nearest")
    return mask


def align_reliable_bg_mask(m_rel_bg: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return align_mask_nearest(m_rel_bg, feat)


def align_reliable_fg_mask(m_rel_fg: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return align_mask_nearest(m_rel_fg, feat)


def align_reliable_valid_mask(m_valid: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return align_mask_nearest(m_valid, feat)


def masked_global_pool(
    feat: torch.Tensor,
    region_mask: torch.Tensor,
    eps: float = 1e-6,
    *,
    align_to_feat: bool = True,
) -> torch.Tensor:
    mask = align_mask_nearest(region_mask, feat) if align_to_feat else region_mask
    denom = mask.sum(dim=(2, 3)).clamp_min(eps)
    pooled = (feat * mask).sum(dim=(2, 3)) / denom
    return pooled


def masked_global_pool_bg(
    bg_feat: torch.Tensor,
    m_rel_bg: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return masked_global_pool(bg_feat, align_reliable_bg_mask(m_rel_bg, bg_feat), eps=eps)


def masked_global_pool_fg(
    fg_feat: torch.Tensor,
    m_rel_fg: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return masked_global_pool(fg_feat, align_reliable_fg_mask(m_rel_fg, fg_feat), eps=eps)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True)
    xy = x @ y.t()
    dist = xx + yy.t() - 2.0 * xy
    gamma = 1.0 / max(2.0 * sigma * sigma, 1e-6)
    return torch.exp(-gamma * dist.clamp_min(0.0))


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_zeros(())
    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)
    m = x.size(0)
    n = y.size(0)
    return k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2.0 * k_xy.sum() / (m * n)


def multi_kernel_gaussian_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    if sigmas is None:
        sigmas = DEFAULT_MMD_SIGMAS
    if not sigmas:
        return gaussian_mmd(x, y, sigma=1.0)
    total = x.new_zeros(())
    for sigma in sigmas:
        total = total + gaussian_mmd(x, y, sigma=float(sigma))
    return total / len(sigmas)


def background_semantic_loss(
    bg_logits: list[torch.Tensor],
    gt_mask: torch.Tensor,
    radius: int = 5,
    lambda_dice: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    fg = _to_fg_mask(gt_mask)
    target_bg = 1.0 - fg
    _m_bg, _m_fg, m_valid = make_reliable_supervision_masks(gt_mask, radius=radius)

    loss = 0.0
    for logit in bg_logits:
        valid = align_reliable_valid_mask(m_valid, logit)
        tgt = align_mask_nearest(target_bg, logit)
        bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        l_bce = (bce * valid).sum() / (valid.sum() + eps)
        prob = torch.sigmoid(logit)
        inter = (prob * tgt * valid).sum(dim=(2, 3))
        union = (prob * valid).sum(dim=(2, 3)) + (tgt * valid).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + eps) / (union + eps)
        loss = loss + l_bce + float(lambda_dice) * dice.mean()
    return loss / max(len(bg_logits), 1)


@torch.no_grad()
def background_semantic_dice(
    bg_logits: list[torch.Tensor],
    gt_mask: torch.Tensor,
    radius: int = 5,
    eps: float = 1e-6,
) -> float:
    fg = _to_fg_mask(gt_mask)
    target = 1.0 - fg
    _m_bg, _m_fg, m_valid = make_reliable_supervision_masks(gt_mask, radius=radius)

    dice_sum = 0.0
    n = 0
    for logit in bg_logits:
        valid = align_reliable_valid_mask(m_valid, logit)
        tgt = align_mask_nearest(target, logit)
        prob = torch.sigmoid(logit)
        inter = (prob * tgt * valid).sum(dim=(2, 3))
        union = (prob * valid).sum(dim=(2, 3)) + (tgt * valid).sum(dim=(2, 3))
        dice = ((2.0 * inter + eps) / (union + eps)).mean()
        dice_sum += float(dice.cpu())
        n += 1
    return dice_sum / max(n, 1)


def foreground_background_allocation_loss(
    m_fg_list: list[torch.Tensor],
    m_bg_list: list[torch.Tensor],
    gt_mask: torch.Tensor,
    radius: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    fg = _to_fg_mask(gt_mask)
    _m_bg, _m_fg, m_valid = make_reliable_supervision_masks(gt_mask, radius=radius)

    loss = 0.0
    n = 0
    for m_fg, m_bg in zip(m_fg_list, m_bg_list):
        valid = align_reliable_valid_mask(m_valid, m_fg)
        tgt_fg = align_mask_nearest(fg, m_fg)
        tgt_bg = align_mask_nearest(1.0 - fg, m_bg)
        pred = torch.cat([m_fg, m_bg], dim=1).clamp(min=eps, max=1.0 - eps)
        target = torch.cat([tgt_fg, tgt_bg], dim=1)
        ce = -(target * torch.log(pred)).sum(dim=1, keepdim=True)
        loss = loss + (ce * valid).sum() / (valid.sum() + eps)
        n += 1
    return loss / max(n, 1)


def background_invariance_mmd_loss(
    bg_feat_list: list[torch.Tensor],
    domain_ids: torch.Tensor,
    region_mask: torch.Tensor,
    n_domains: int = 2,
    mmd_sigmas: tuple[float, ...] | list[float] | None = None,
    layer_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if domain_ids.dim() != 1:
        domain_ids = domain_ids.view(-1)
    if layer_indices is None:
        layer_indices = MMD_INV_LAYER_INDICES

    total = bg_feat_list[0].new_zeros(())
    n_terms = 0
    for idx in layer_indices:
        if idx < 0 or idx >= len(bg_feat_list):
            continue
        bg_feat = bg_feat_list[idx]
        z = masked_global_pool_bg(bg_feat, region_mask)
        z = F.normalize(z, dim=1)
        for a in range(n_domains):
            za = z[domain_ids == a]
            for b in range(a + 1, n_domains):
                zb = z[domain_ids == b]
                if za.size(0) >= 2 and zb.size(0) >= 2:
                    total = total + multi_kernel_gaussian_mmd(za, zb, sigmas=mmd_sigmas)
                    n_terms += 1
    if n_terms == 0:
        return total
    return total / n_terms


def compute_bgf_losses(
    bg_feat_list: list[torch.Tensor],
    bg_logit_list: list[torch.Tensor],
    gt_mask: torch.Tensor,
    domain_ids: torch.Tensor | None = None,
    m_fg_list: list[torch.Tensor] | None = None,
    m_bg_list: list[torch.Tensor] | None = None,
    bg_radius: int = 5,
    mmd_sigmas: tuple[float, ...] | list[float] | None = None,
    lambda_dice: float = 1.0,
    n_domains: int = 2,
) -> dict[str, torch.Tensor]:
    m_bg, _m_fg, _m_valid = make_reliable_supervision_masks(gt_mask, radius=bg_radius)

    sem_loss = background_semantic_loss(
        bg_logit_list, gt_mask, radius=bg_radius, lambda_dice=lambda_dice,
    )

    inv_loss = bg_feat_list[0].new_zeros(())
    if domain_ids is not None:
        inv_loss = background_invariance_mmd_loss(
            bg_feat_list, domain_ids, m_bg,
            n_domains=n_domains, mmd_sigmas=mmd_sigmas,
        )

    mask_loss = bg_feat_list[0].new_zeros(())
    if m_fg_list is not None and m_bg_list is not None:
        mask_loss = foreground_background_allocation_loss(
            m_fg_list, m_bg_list, gt_mask, radius=bg_radius,
        )

    return {
        "bg_sem": sem_loss,
        "inv": inv_loss,
        "mask": mask_loss,
    }


@torch.no_grad()
def pool_domain_mmd_features(
    bg_feat_list: list[torch.Tensor],
    fg_feat_list: list[torch.Tensor],
    m_rel_bg: torch.Tensor,
    m_rel_fg: torch.Tensor,
    layer_index: int = DEFAULT_MMD_LAYER_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer_index = min(max(layer_index, 0), len(bg_feat_list) - 1)
    z_bg = masked_global_pool_bg(bg_feat_list[layer_index], m_rel_bg)
    fg_list = fg_feat_list if fg_feat_list else bg_feat_list
    z_fg = masked_global_pool_fg(fg_list[layer_index], m_rel_fg)
    return z_bg, z_fg


@torch.no_grad()
def pooled_feature_variance(vecs_by_domain: dict[int, list[torch.Tensor]]) -> float | None:
    chunks = []
    for domain_vecs in vecs_by_domain.values():
        if domain_vecs:
            chunks.append(torch.cat(domain_vecs, dim=0))
    if not chunks:
        return None
    all_z = torch.cat(chunks, dim=0)
    if all_z.size(0) < 2:
        return None
    return float(all_z.var(dim=0, unbiased=False).mean().cpu())


@torch.no_grad()
def finalize_domain_mmd_metrics(
    bg_vecs_by_domain: dict[int, list[torch.Tensor]],
    fg_vecs_by_domain: dict[int, list[torch.Tensor]],
    mmd_sigmas: tuple[float, ...] | list[float] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, buckets in (("mmd_bg", bg_vecs_by_domain), ("mmd_fg", fg_vecs_by_domain)):
        if 0 not in buckets or 1 not in buckets:
            continue
        if len(buckets[0]) < 2 or len(buckets[1]) < 2:
            continue
        za = F.normalize(torch.cat(buckets[0], dim=0), dim=1)
        zb = F.normalize(torch.cat(buckets[1], dim=0), dim=1)
        if za.size(0) >= 2 and zb.size(0) >= 2:
            metrics[key] = float(
                multi_kernel_gaussian_mmd(za, zb, sigmas=mmd_sigmas).cpu()
            )
    var_bg = pooled_feature_variance(bg_vecs_by_domain)
    if var_bg is not None:
        metrics["var_bg"] = var_bg
    return metrics
