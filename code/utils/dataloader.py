import json
import os
import random
import re

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

torch.backends.cudnn.enabled = False

SUBSET_KVASIR = "kvasir"
SUBSET_CVC_CLINIC = "cvc_clinic"
TRAIN_SUBSETS = (SUBSET_KVASIR, SUBSET_CVC_CLINIC)
DOMAIN_KVASIR = 0
DOMAIN_CVC_CLINIC = 1


def infer_train_subset(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if re.fullmatch(r"\d+", base):
        return SUBSET_CVC_CLINIC
    return SUBSET_KVASIR


def subset_to_domain_id(source_dataset: str) -> int:
    return DOMAIN_KVASIR if source_dataset == SUBSET_KVASIR else DOMAIN_CVC_CLINIC


def build_train_manifest(image_root: str, gt_root: str) -> list[dict]:
    pairs = _list_image_mask_pairs(image_root, gt_root)
    manifest = []
    for img_path, mask_path in pairs:
        source = infer_train_subset(img_path)
        manifest.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "source_dataset": source,
            "domain_id": subset_to_domain_id(source),
        })
    return manifest


def assign_manifest_splits(
    manifest: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    if len(manifest) < 2:
        raise ValueError(f"样本数过少，无法划分验证集: n={len(manifest)}")

    by_subset = {name: [] for name in TRAIN_SUBSETS}
    for idx, rec in enumerate(manifest):
        by_subset[rec["source_dataset"]].append(idx)

    rng = np.random.RandomState(seed)
    subset_stats = {}
    out = [dict(rec) for rec in manifest]

    for subset in TRAIN_SUBSETS:
        indices = by_subset[subset]
        if not indices:
            continue
        n = len(indices)
        n_val = max(1, int(round(n * val_ratio)))
        if n_val >= n:
            n_val = n - 1
        perm = rng.permutation(n)
        val_local = {indices[i] for i in perm[:n_val]}
        train_local = {indices[i] for i in perm[n_val:]}
        for idx in indices:
            out[idx]["split"] = "val" if idx in val_local else "train"
        subset_stats[subset] = {
            "total": n,
            "train": len(train_local),
            "val": len(val_local),
        }

    return out, subset_stats


def save_manifest(manifest: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _list_image_mask_pairs(image_root, gt_root):
    images = sorted([
        os.path.join(image_root, f)
        for f in os.listdir(image_root)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    gts = sorted([
        os.path.join(gt_root, f)
        for f in os.listdir(gt_root)
        if f.endswith(".png")
    ])
    pairs = []
    for img_path, gt_path in zip(images, gts):
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        if img.size == gt.size:
            pairs.append((img_path, gt_path))
    return pairs


def save_val_split(split_info, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)


class PolypDataset(data.Dataset):
    def __init__(self, trainsize=352, records=None, augmentation=False):
        if not records:
            raise ValueError("records is required")
        self.trainsize = trainsize
        self.augmentation = augmentation
        self.records = list(records)
        self.pairs = [(r["image_path"], r["mask_path"]) for r in self.records]
        self.domain_ids = [int(r["domain_id"]) for r in self.records]
        self.size = len(self.pairs)
        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01,
                ),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        img_path, gt_path = self.pairs[index]
        image = self.rgb_loader(img_path)
        gt = self.binary_loader(gt_path)
        if self.augmentation:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.img_transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            gt = self.gt_transform(gt)
        else:
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
        return image, gt, self.domain_ids[index]

    @staticmethod
    def rgb_loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    @staticmethod
    def binary_loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):
        return self.size


class BalancedDomainBatchSampler(data.Sampler):
    def __init__(self, dataset, batch_size: int, seed: int = 42):
        if batch_size % 2 != 0:
            raise ValueError(
                f"BalancedDomainBatchSampler 要求偶数 batch_size，当前={batch_size}",
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = int(seed)
        self.epoch = 0

        k_indices: list[int] = []
        c_indices: list[int] = []
        for idx in range(len(dataset)):
            domain_id = dataset.domain_ids[idx]
            if domain_id == DOMAIN_KVASIR:
                k_indices.append(idx)
            else:
                c_indices.append(idx)
        if not k_indices or not c_indices:
            raise ValueError("训练集需同时包含 Kvasir 与 CVC-ClinicDB 样本")
        self.k_indices = k_indices
        self.c_indices = c_indices

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        n = min(len(self.k_indices), len(self.c_indices))
        n = (n // self.half) * self.half
        return max(n // self.half, 0)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        k = rng.permutation(self.k_indices).tolist()
        c = rng.permutation(self.c_indices).tolist()
        n = min(len(k), len(c))
        n = (n // self.half) * self.half
        k = k[:n]
        c = c[:n]
        for i in range(0, n, self.half):
            batch = k[i:i + self.half] + c[i:i + self.half]
            rng.shuffle(batch)
            yield batch


def _build_train_val_loaders(
    train_dataset,
    val_dataset,
    batchsize,
    num_workers,
    pin_memory,
    balanced_domain_batch=True,
    seed=42,
):
    if balanced_domain_batch:
        batch_sampler = BalancedDomainBatchSampler(
            train_dataset, batchsize, seed=seed,
        )
        train_loader = data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_train_val_loaders(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    val_ratio=0.1,
    seed=42,
    num_workers=4,
    pin_memory=True,
    augmentation=True,
    balanced_domain_batch=True,
):
    manifest = build_train_manifest(image_root, gt_root)
    manifest, subset_stats = assign_manifest_splits(
        manifest, val_ratio=val_ratio, seed=seed,
    )
    train_records = [rec for rec in manifest if rec.get("split") == "train"]
    val_records = [rec for rec in manifest if rec.get("split") == "val"]
    val_policy = {
        SUBSET_KVASIR: f"{val_ratio:.0%}_holdout",
        SUBSET_CVC_CLINIC: f"{val_ratio:.0%}_holdout",
    }

    train_dataset = PolypDataset(
        trainsize=trainsize,
        records=train_records,
        augmentation=augmentation,
    )
    val_dataset = PolypDataset(
        trainsize=trainsize,
        records=val_records,
        augmentation=False,
    )
    train_loader, val_loader = _build_train_val_loaders(
        train_dataset, val_dataset, batchsize, num_workers, pin_memory,
        balanced_domain_batch=balanced_domain_batch,
        seed=seed,
    )
    split_info = {
        "seed": seed,
        "val_ratio": val_ratio,
        "val_source": "TrainDataset",
        "val_policy": val_policy,
        "train_path": os.path.dirname(image_root.rstrip(os.sep)),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "total_samples": len(manifest),
        "subset_stats": subset_stats,
        "manifest": manifest,
        "train_files": [os.path.basename(rec["image_path"]) for rec in train_records],
        "val_files": [os.path.basename(rec["image_path"]) for rec in val_records],
        "train_pairs": [
            {"image": rec["image_path"], "mask": rec["mask_path"]}
            for rec in train_records
        ],
        "val_pairs": [
            {"image": rec["image_path"], "mask": rec["mask_path"]}
            for rec in val_records
        ],
    }
    return train_loader, val_loader, split_info


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
