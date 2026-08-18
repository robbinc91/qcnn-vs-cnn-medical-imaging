"""HAMMERS parietal-lobe segmentation dataset.

File naming: a??.nii.gz (T1, int16 -> float32) and a??-seg.nii.gz (labels).
Labels remapped: {32,33,60,61,62,63,84,85} -> {1..8}.

Supports both 2-D (axial slice) and 3-D (full volume) training.
"""

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

NUM_CLASSES = 9
CLASS_NAMES = [
    "background",
    "left_superior_parietal", "right_superior_parietal",
    "left_supramarginal", "right_supramarginal",
    "left_angular", "right_angular",
    "left_postcentral", "right_postcentral",
]
HAMMERS_LABELS = [32, 33, 60, 61, 62, 63, 84, 85]
LABEL_MAP = {v: i + 1 for i, v in enumerate(HAMMERS_LABELS)}

_ALL_IDS = list(range(1, 31))


def get_subject_splits(seed: int = 42) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(_ALL_IDS))
    ids = np.array(_ALL_IDS)[perm].tolist()
    return {
        "test":  sorted(ids[:5]),
        "val":   sorted(ids[5:10]),
        "train": sorted(ids[10:]),
    }


def _remap_labels(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int32)
    for src, dst in LABEL_MAP.items():
        out[seg == src] = dst
    return out


# Per-subject label-bbox crop: union bbox (158, 128, 98) over all 30 subjects + 6-vox margin.
CROP_H, CROP_W, CROP_D = 164, 134, 104
CROP_MARGIN = 6


def _bbox_crop(vol: np.ndarray, seg: np.ndarray,
               target_h: int = CROP_H, target_w: int = CROP_W,
               target_d: Optional[int] = CROP_D, margin: int = CROP_MARGIN) -> Tuple[np.ndarray, np.ndarray]:
    """Crop vol/seg to a fixed-size box centered on the label bounding box (+margin).

    Guaranteed to contain every label voxel: the target size is the union bbox over
    all subjects plus margin, so each subject's bbox fits with >= 0 margin. Pass
    target_d=None to leave the depth axis at bbox +/- margin (2D slice loader).
    """
    fg = np.argwhere(seg > 0)
    mn = fg.min(axis=0)
    mx = fg.max(axis=0)
    lo, hi = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    for axis, t in enumerate((target_h, target_w, vol.shape[2] if target_d is None else target_d)):
        if t is None:
            lo[axis] = max(0, int(mn[axis]) - margin)
            hi[axis] = min(vol.shape[axis] - 1, int(mx[axis]) + margin)
            continue
        t = int(t)
        bsize = int(mx[axis] - mn[axis]) + 1
        m = min(margin, (t - bsize) // 2)
        box_lo = int(mn[axis]) - m
        box_hi = box_lo + t - 1
        if box_lo < 0:
            box_lo, box_hi = 0, t - 1
        elif box_hi > vol.shape[axis] - 1:
            box_hi = vol.shape[axis] - 1
            box_lo = box_hi - t + 1
        lo[axis], hi[axis] = box_lo, box_hi
    return (vol[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1],
            seg[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1])


class HammersAugmentation:
    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.train:
            return img, mask
        mask_f = mask.unsqueeze(0).float()
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask_f = TF.hflip(mask_f)
        if random.random() < 0.25:
            img = TF.vflip(img)
            mask_f = TF.vflip(mask_f)
        angle = random.uniform(-15.0, 15.0)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        mask_f = TF.rotate(mask_f, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        if random.random() < 0.5:
            tx = random.uniform(-0.08, 0.08) * img.shape[-1]
            ty = random.uniform(-0.08, 0.08) * img.shape[-2]
            scale = random.uniform(0.90, 1.10)
            img = TF.affine(img, angle=0, translate=[tx, ty], scale=scale, shear=0,
                            interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            mask_f = TF.affine(mask_f, angle=0, translate=[tx, ty], scale=scale, shear=0,
                               interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        if random.random() < 0.5:
            img = img + torch.randn_like(img) * 0.05
        if random.random() < 0.3:
            img = img * random.uniform(0.90, 1.10)
        if random.random() < 0.2:
            h, w = img.shape[-2], img.shape[-1]
            bias = torch.randn(1, 1, 2, 2) * 0.05
            bias = F.interpolate(bias, size=(h, w), mode="bilinear", align_corners=False)
            img = img + bias.squeeze(0)
        mask = mask_f.squeeze(0).round().long()
        return img, mask


class HammersVolumeAugmentation:
    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(self, vol: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.train:
            return vol, seg
        C, H, W, D = vol.shape
        angle_deg = random.uniform(-15.0, 15.0)
        tx_n, ty_n, scale = 0.0, 0.0, 1.0
        if random.random() < 0.5:
            tx_n = random.uniform(-0.08, 0.08) * 2.0
            ty_n = random.uniform(-0.08, 0.08) * 2.0
            scale = random.uniform(0.90, 1.10)
        vol, seg = _axial_affine(vol, seg, angle_deg, tx_n, ty_n, scale)
        if random.random() < 0.5:
            vol = vol + torch.randn_like(vol) * 0.05
        if random.random() < 0.3:
            vol = vol * random.uniform(0.90, 1.10)
        if random.random() < 0.2:
            bias = torch.randn(1, 1, 2, 2, 2) * 0.05
            bias = F.interpolate(bias, size=(H, W, D), mode="trilinear", align_corners=False)
            vol = vol + bias.squeeze(0)
        return vol, seg


def _axial_affine(vol: torch.Tensor, seg: torch.Tensor,
                  angle_deg: float, tx_n: float, ty_n: float, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W, D = vol.shape
    a = math.radians(angle_deg)
    ca, sa = math.cos(a) * scale, math.sin(a) * scale
    theta = torch.tensor([[[ca, -sa, tx_n], [sa, ca, ty_n]]], dtype=torch.float32).expand(D, -1, -1)
    grid = F.affine_grid(theta, (D, 1, H, W), align_corners=False)
    vol_in = vol.permute(3, 0, 1, 2)
    vol_out = F.grid_sample(vol_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    vol_out = vol_out.permute(1, 2, 3, 0)
    seg_in = seg.permute(2, 0, 1).unsqueeze(1).float()
    seg_out = F.grid_sample(seg_in, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    seg_out = seg_out.squeeze(1).permute(1, 2, 0).long()
    return vol_out, seg_out


class HammersSliceDataset(Dataset):
    def __init__(self, data_dir: str, subject_ids: List[int],
                 augmentation: Optional[HammersAugmentation] = None,
                 min_foreground_voxels: int = 20) -> None:
        self.data_dir = Path(data_dir)
        self.augmentation = augmentation
        self.min_fg = min_foreground_voxels
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self._load_subjects(subject_ids)
        fg_voxels = sum(int((m > 0).sum()) for m in self.masks)
        logger.info(f"HammersSliceDataset: {len(self.masks)} slices from {len(subject_ids)} subjects | fg voxels: {fg_voxels:,}")

    def _load_subjects(self, subject_ids: List[int]) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel required: pip install nibabel") from exc
        for sid in subject_ids:
            img_path = self.data_dir / f"a{sid:02d}.nii.gz"
            seg_path = self.data_dir / f"a{sid:02d}-seg.nii.gz"
            if not img_path.exists() or not seg_path.exists():
                logger.warning(f"Missing files for a{sid:02d}, skipping.")
                continue
            try:
                volume = nib.load(str(img_path)).get_fdata(dtype=np.float32)
                seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
            except Exception as exc:
                logger.warning(f"Failed to load a{sid:02d}: {exc}")
                continue
            seg = _remap_labels(seg)
            volume, seg = _bbox_crop(volume, seg, target_d=None)
            mean, std = float(volume.mean()), float(volume.std()) + 1e-8
            volume = (volume - mean) / std
            H, W, D = volume.shape
            for z in range(D):
                seg_slice = seg[:, :, z]
                if int((seg_slice > 0).sum()) < self.min_fg:
                    continue
                self.images.append(volume[:, :, z].copy())
                self.masks.append(seg_slice.copy())

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        mask = torch.from_numpy(self.masks[idx]).long()
        if self.augmentation is not None:
            img, mask = self.augmentation(img, mask)
        return img, mask

    def class_weights(self) -> torch.Tensor:
        all_labels = np.concatenate([m.ravel() for m in self.masks])
        counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (NUM_CLASSES * counts)
        return torch.from_numpy(weights)


class HammersVolumeDataset(Dataset):
    def __init__(self, data_dir: str, subject_ids: List[int],
                 augmentation: Optional[HammersVolumeAugmentation] = None) -> None:
        self.augmentation = augmentation
        self.volumes: List[np.ndarray] = []
        self.segs: List[np.ndarray] = []
        self._load_subjects(Path(data_dir), subject_ids)
        logger.info(f"HammersVolumeDataset: {len(self.volumes)} volumes from {len(subject_ids)} subjects")

    def _load_subjects(self, data_dir: Path, subject_ids: List[int]) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel required: pip install nibabel") from exc
        for sid in subject_ids:
            img_path = data_dir / f"a{sid:02d}.nii.gz"
            seg_path = data_dir / f"a{sid:02d}-seg.nii.gz"
            if not img_path.exists() or not seg_path.exists():
                logger.warning(f"Missing a{sid:02d}, skipping.")
                continue
            try:
                vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
                seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
            except Exception as exc:
                logger.warning(f"Failed to load a{sid:02d}: {exc}")
                continue
            seg = _remap_labels(seg)
            vol, seg = _bbox_crop(vol, seg)
            mean, std = float(vol.mean()), float(vol.std()) + 1e-8
            self.volumes.append(((vol - mean) / std).copy())
            self.segs.append(seg.copy())

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol = torch.from_numpy(self.volumes[idx]).unsqueeze(0)
        seg = torch.from_numpy(self.segs[idx]).long()
        if self.augmentation is not None:
            vol, seg = self.augmentation(vol, seg)
        return vol, seg

    def class_weights(self) -> torch.Tensor:
        all_labels = np.concatenate([s.ravel() for s in self.segs])
        counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (NUM_CLASSES * counts)
        return torch.from_numpy(weights)


def HammersDataFactory(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    ds_cfg = cfg.dataset
    splits = get_subject_splits(seed=cfg.get("seed", 42))
    min_fg = ds_cfg.get("min_foreground_voxels", 20)
    data_dir = ds_cfg.data_dir
    train_aug = HammersAugmentation(train=True)
    eval_aug = HammersAugmentation(train=False)
    train_ds = HammersSliceDataset(data_dir, splits["train"], train_aug, min_fg)
    val_ds = HammersSliceDataset(data_dir, splits["val"], eval_aug, min_fg)
    test_ds = HammersSliceDataset(data_dir, splits["test"], eval_aug, min_fg)
    nw = ds_cfg.get("num_workers", 0)
    bs = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    logger.info(f"2D Hammers -> train: {len(train_ds)} slices ({len(splits['train'])} subjects) | val: {len(val_ds)} ({len(splits['val'])}) | test: {len(test_ds)} ({len(splits['test'])}")
    return train_loader, val_loader, test_loader, train_ds.class_weights()


def HammersVolumeDataFactory(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    ds_cfg = cfg.dataset
    splits = get_subject_splits(seed=cfg.get("seed", 42))
    data_dir = ds_cfg.data_dir
    train_aug = HammersVolumeAugmentation(train=True)
    eval_aug = HammersVolumeAugmentation(train=False)
    train_ds = HammersVolumeDataset(data_dir, splits["train"], train_aug)
    val_ds = HammersVolumeDataset(data_dir, splits["val"], eval_aug)
    test_ds = HammersVolumeDataset(data_dir, splits["test"], eval_aug)
    nw = ds_cfg.get("num_workers", 0)
    bs = cfg.training.get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    logger.info(f"3D Hammers -> train: {len(train_ds)} vols | val: {len(val_ds)} | test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.class_weights()
