import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GT_COLOR = np.array([0, 200, 0], dtype=np.float32)
PRED_COLOR = np.array([255, 50, 50], dtype=np.float32)
OVERLAP_COLOR = np.array([255, 220, 0], dtype=np.float32)
SEG_HEADS = (('p2', 'p2'), ('p3', 'p3'), ('p4', 'p4'), ('lout', 'lout'))
PREDICTION_HEADS = tuple((key for _, key in SEG_HEADS))
ENC_HEAT_KEYS = ('e1', 'e2', 'e3', 'e4', 'e5_pd')
DEC_HEAT_KEYS = ('d4', 'd3', 'd2', 'd1')
BG_HEAT_KEYS = ('bg4', 'bg3', 'bg2')
FG_HEAT_KEYS = ('fg4', 'fg3', 'fg2')
FG_SUPP_HEAT_KEYS = ('fg4_suppressed', 'fg3_suppressed', 'fg2_suppressed')
BG_LOGIT_HEAT_KEYS = ('bg_logit4', 'bg_logit3', 'bg_logit2')
M_FG_HEAT_KEYS = ('m_fg4', 'm_fg3', 'm_fg2')
M_BG_HEAT_KEYS = ('m_bg4', 'm_bg3', 'm_bg2')
HEAT_KEYS = ENC_HEAT_KEYS + DEC_HEAT_KEYS + BG_HEAT_KEYS + FG_HEAT_KEYS + FG_SUPP_HEAT_KEYS + BG_LOGIT_HEAT_KEYS + M_FG_HEAT_KEYS + M_BG_HEAT_KEYS
def denormalize_image(image_tensor):
    img = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)
def pil_mask_to_array(gt_pil, size):
    gt = gt_pil.convert('L').resize(size, Image.BILINEAR)
    arr = np.asarray(gt, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr
def logit_to_prob(logit, size):
    prob = torch.sigmoid(logit)
    if prob.shape[-2:] != size[::-1]:
        prob = F.interpolate(prob, size=size, mode='bilinear', align_corners=False)
    prob = prob.squeeze().detach().cpu().numpy()
    return np.clip(prob, 0.0, 1.0)
def binary_mask(prob, threshold=0.5):
    return prob >= threshold
def overlay_gt_pred(image_rgb, gt_mask, pred_mask, alpha=0.45):
    base = image_rgb.astype(np.float32)
    out = base.copy()
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    overlap = gt & pred
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    def _blend(region, color):
        if not region.any():
            return
        out[region] = (1.0 - alpha) * base[region] + alpha * color
    _blend(gt_only, GT_COLOR)
    _blend(pred_only, PRED_COLOR)
    _blend(overlap, OVERLAP_COLOR)
    return np.clip(out, 0, 255).astype(np.uint8)
def overlay_gt_only(image_rgb, gt_mask, alpha=0.45):
    base = image_rgb.astype(np.float32)
    out = base.copy()
    region = gt_mask.astype(bool)
    if region.any():
        out[region] = (1.0 - alpha) * base[region] + alpha * GT_COLOR
    return np.clip(out, 0, 255).astype(np.uint8)
def _draw_seg_legend(ax):
    ax.axis('off')
    items = [('GT region', GT_COLOR), ('Pred region', PRED_COLOR), ('Overlap', OVERLAP_COLOR)]
    y = 0.88
    for label, color in items:
        ax.add_patch(plt.Rectangle((0.06, y - 0.06), 0.16, 0.09, color=color / 255.0, transform=ax.transAxes))
        ax.text(0.26, y - 0.01, label, transform=ax.transAxes, fontsize=10, va='center')
        y -= 0.24
    ax.text(0.06, 0.06, 'Seg heads: region overlay', transform=ax.transAxes, fontsize=9, va='bottom')
def feature_to_heatmap(feat, size):
    if feat.dim() == 4:
        heat = feat.detach().float().abs().mean(dim=1, keepdim=True)
    elif feat.dim() == 3:
        heat = feat.detach().float().abs().unsqueeze(0)
    else:
        heat = feat.detach().float().abs()
    heat = F.interpolate(heat, size=size, mode='bilinear', align_corners=False)
    heat = heat.squeeze().cpu().numpy()
    heat = heat - heat.min()
    peak = heat.max()
    if peak > 1e-08:
        heat = heat / peak
    return heat.astype(np.float32)
def overlay_heatmap(image_rgb, heat, alpha=0.55, cmap_name='jet'):
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(np.clip(heat, 0.0, 1.0))[..., :3]
    colored = (colored * 255.0).astype(np.float32)
    base = image_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * colored
    return np.clip(out, 0, 255).astype(np.uint8)
def save_all_heads_overview(image_tensor, gt_pil, model_outputs, save_path, sample_name='', threshold=0.5):
    display_h, display_w = (image_tensor.shape[-2], image_tensor.shape[-1])
    rgb = denormalize_image(image_tensor)
    gt_prob = pil_mask_to_array(gt_pil, (display_w, display_h))
    gt_bin = binary_mask(gt_prob, threshold)
    head_probs = {}
    for name in PREDICTION_HEADS:
        head_probs[name] = logit_to_prob(model_outputs[name], (display_h, display_w))
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f'Prediction Heads — {sample_name}' if sample_name else 'Prediction Heads', fontsize=14, y=0.98)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Image')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(overlay_gt_only(rgb, gt_bin))
    axes[0, 1].set_title('GT')
    axes[0, 1].axis('off')
    _draw_seg_legend(axes[0, 2])
    axes[0, 2].set_title('Legend')
    for idx, (label, key) in enumerate(SEG_HEADS):
        r, c = divmod(idx, 3)
        ax = axes[r + 1, c]
        pred_bin = binary_mask(head_probs[key], threshold)
        ax.imshow(overlay_gt_pred(rgb, gt_bin, pred_bin))
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
def _stem(name):
    return os.path.splitext(name)[0]
def _save_rgb(path, rgb):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(rgb).save(path)
def save_feature_heatmaps_separate(image_tensor, feats, save_dir, sample_name=''):
    display_h, display_w = (image_tensor.shape[-2], image_tensor.shape[-1])
    size = (display_h, display_w)
    rgb = denormalize_image(image_tensor)
    stem = _stem(sample_name) if sample_name else 'sample'
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    for key in HEAT_KEYS:
        if key not in feats:
            continue
        heat = feature_to_heatmap(feats[key], size)
        path = os.path.join(save_dir, f'{stem}_{key}.png')
        _save_rgb(path, overlay_heatmap(rgb, heat))
        saved.append(path)
    return saved
def export_dataset_head_overview(model, dataset_name, data_root, save_dir, testsize, device, sample_index=0):
    from utils.dataloader import test_dataset
    data_path = os.path.join(data_root, dataset_name)
    image_root = os.path.join(data_path, 'images') + os.sep
    gt_root = os.path.join(data_path, 'masks') + os.sep
    loader = test_dataset(image_root, gt_root, testsize)
    if loader.size == 0:
        return (None, [])
    sample_index = min(sample_index, loader.size - 1)
    loader.index = sample_index
    image, gt_pil, name = loader.load_data()
    with torch.no_grad():
        outputs, feats = model(image.to(device), return_feats=True)
    os.makedirs(save_dir, exist_ok=True)
    heads_path = os.path.join(save_dir, f'{dataset_name}_all_heads.png')
    save_all_heads_overview(image, gt_pil, outputs, heads_path, sample_name=name)
    feat_dir = os.path.join(save_dir, dataset_name, 'feat_heatmaps')
    feat_paths = save_feature_heatmaps_separate(image, feats, feat_dir, sample_name=name)
    return (heads_path, feat_paths)