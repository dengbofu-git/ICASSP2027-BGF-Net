import os

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from eval import eval_for_testAllInOne
from lib.baseline import BaseModel
from utils.dataloader import test_dataset
from utils.utils import format_metrics, round_metric
from utils.visualize import export_dataset_head_overview

torch.backends.cudnn.enabled = False

TEST_METRICS = ["meanDic", "meanIoU", "wFm", "Sm", "maxEm", "mae"]


def discover_datasets(data_root, datasets):
    if datasets:
        return datasets
    names = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "images")):
            names.append(name)
    return names


def load_model(weight_path, device, fg_keep_eps=0.05):
    state = torch.load(weight_path, map_location="cpu")
    net = BaseModel(
        num_class=1,
        pretrained=False,
        fg_keep_eps=fg_keep_eps,
    )
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    return net


def predict_lout(model, image, gt_shape):
    outputs = model(image)
    l_out = outputs["lout"]
    l_out = F.interpolate(l_out, size=gt_shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(l_out).squeeze().detach().cpu().numpy()


def infer_and_eval(
    model, dataset_name, data_root, save_root, testsize, metrics, device,
):
    data_path = os.path.join(data_root, dataset_name)
    image_root = os.path.join(data_path, "images") + os.sep
    gt_root = os.path.join(data_path, "masks") + os.sep
    save_path = os.path.join(save_root, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    loader = test_dataset(image_root, gt_root, testsize)
    per_image = np.zeros((loader.size, len(metrics)))

    for idx in tqdm(range(loader.size), desc=f"{dataset_name}/lout", leave=False):
        image, gt, name = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= gt.max() + 1e-8
        image = image.to(device)

        pred = predict_lout(model, image, gt.shape)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_uint8 = (pred * 255).astype(np.uint8)

        eval_config = {"metrics": metrics, "test_size": testsize}
        per_image[idx, :] = eval_for_testAllInOne(eval_config, pred_uint8, gt)

        import imageio
        imageio.imwrite(os.path.join(save_path, name), pred_uint8)

    return np.array([round_metric(x) for x in np.mean(per_image, axis=0)])


def run_lout_eval(
    model, datasets, data_root, save_root, result_path, testsize, metrics, device,
):
    os.makedirs(result_path, exist_ok=True)
    vis_dir = os.path.join(result_path, "head_vis")
    os.makedirs(vis_dir, exist_ok=True)
    for dataset_name in datasets:
        heads_path, feat_paths = export_dataset_head_overview(
            model, dataset_name, data_root, vis_dir, testsize, device,
        )
        if heads_path:
            print(f"[visualize] {dataset_name} prediction heads -> {heads_path}")
        if feat_paths:
            feat_dir = os.path.dirname(feat_paths[0])
            print(
                f"[visualize] {dataset_name} feature heatmaps "
                f"({len(feat_paths)} files) -> {feat_dir}"
            )

    rows = []
    for dataset_name in datasets:
        scores = infer_and_eval(
            model, dataset_name, data_root, save_root,
            testsize, metrics, device,
        )
        rows.append([dataset_name, *scores.tolist()])

        csv_path = os.path.join(result_path, f"result_{dataset_name}_lout.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("method," + ",".join(metrics) + "\n")
            f.write("lout," + ",".join(format_metrics(scores)) + "\n")

    summary_path = os.path.join(result_path, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("dataset," + ",".join(metrics) + "\n")
        for row in rows:
            f.write(row[0] + "," + ",".join(format_metrics(row[1:])) + "\n")

    table = tabulate(rows, headers=["dataset", *metrics], floatfmt=".3f")
    print("\n" + "#" * 20 + " Evaluation (lout) " + "#" * 20)
    print(table)
    print(f"\nPredictions: {save_root}")
    print(f"CSV results: {result_path}")
    print(f"Head visualization: {vis_dir}")
    print(f"Summary: {summary_path}")
    return rows
