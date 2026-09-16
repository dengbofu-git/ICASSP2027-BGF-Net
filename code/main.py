#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from eval_infer import (
    TEST_METRICS,
    discover_datasets,
    load_model,
    run_lout_eval,
)
from train import (
    add_training_args,
    build_optimizer,
    run_training_loop,
)
from lib.baseline import BaseModel, FIXED_PVT_MODEL, format_learnable_weights
from utils.dataloader import get_train_val_loaders, save_manifest, save_val_split
from utils.utils import format_metric, round_metric

torch.backends.cudnn.enabled = False
METRICS = TEST_METRICS


def parse_args():
    parser = argparse.ArgumentParser(description="BGF-Net")
    parser.add_argument("project", type=str)
    add_training_args(parser)
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def build_train_loaders(args, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    image_root = os.path.join(args.train_path, "images")
    gt_root = os.path.join(args.train_path, "masks")
    train_loader, val_loader, split_info = get_train_val_loaders(
        image_root, gt_root,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        val_ratio=getattr(args, "val_ratio", 0.1),
        seed=getattr(args, "seed", 42),
        augmentation=getattr(args, "augmentation", True),
    )
    save_val_split(split_info, os.path.join(save_dir, "val_split.json"))
    if "manifest" in split_info:
        save_manifest(split_info["manifest"], os.path.join(save_dir, "data_manifest.json"))
    return train_loader, val_loader, split_info


def run_training(args):
    print("#" * 20, "开始训练", "#" * 20)
    print(f"项目: {args.project}")
    print(f"骨干: PVTv2-{FIXED_PVT_MODEL.upper()} (fixed)")
    print(f"训练数据: {args.train_path}")
    print(f"设备: {args.device}")

    save_dir = os.path.join("snapshots", args.project)
    train_loader, val_loader, split_info = build_train_loaders(args, save_dir)
    print(
        f"训练样本: {split_info['train_samples']} | "
        f"验证样本: {split_info['val_samples']} "
        f"({split_info['val_source']} {split_info['val_policy']}) | "
        f"子集统计: {split_info['subset_stats']}"
    )

    model = BaseModel(
        num_class=1,
        fg_keep_eps=getattr(args, "fg_keep_eps", 0.05),
    ).to(args.device)
    optimizer = build_optimizer(
        model, args.lr, optimizer_name=getattr(args, "optimizer", "AdamW"),
    )

    class TrainOpt:
        pass

    opt = TrainOpt()
    for key, value in vars(args).items():
        setattr(opt, key, value)
    opt.train_save = args.project

    best_path, best_epoch, best_val_sid = run_training_loop(
        model, optimizer, train_loader, val_loader, opt, save_dir,
    )
    print(f"训练完成，最优 epoch={best_epoch}，S_ID={format_metric(best_val_sid)}")
    return {
        "best_path": best_path,
        "best_epoch": best_epoch,
        "best_val_sid": round_metric(best_val_sid),
    }


def run_inference(args, weight_path):
    print("#" * 20, "开始推理与评估", "#" * 20)
    print(f"权重: {weight_path}")

    datasets = discover_datasets(args.test_path, args.datasets)
    if not datasets:
        raise FileNotFoundError(f"在 {args.test_path} 下未找到测试数据集")

    model = load_model(
        weight_path, args.device,
        fg_keep_eps=getattr(args, "fg_keep_eps", 0.05),
    )
    print(format_learnable_weights(model))
    print("推理输出: lout")

    result_root = os.path.join(ROOT_DIR, "results", args.project)
    eval_result_root = os.path.join(ROOT_DIR, "eval_results", args.project)
    run_lout_eval(
        model, datasets, args.test_path, result_root, eval_result_root,
        args.testsize, METRICS, args.device,
    )
    return result_root


def main():
    args = parse_args()
    print("=" * 60)
    print(f"BGF-Net Pipeline  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    best_path = os.path.join("snapshots", args.project, "best.pth")
    if args.skip_train:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"--skip_train 已指定，但未找到: {best_path}")
        print(f"跳过训练，使用: {best_path}")
    else:
        train_result = run_training(args)
        best_path = train_result["best_path"]

    run_inference(args, best_path)
    print("\n全部流程执行完毕。")


if __name__ == "__main__":
    main()
