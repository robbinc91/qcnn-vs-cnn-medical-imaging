"""3D NIfTI brainstem dataset with 2D axial slice extraction for segmentation.

Strategy: convert 3D volumes to 2D axial slices, returning paired
(image, segmentation_mask) for pixel-level segmentation training.

Labels:  0 = background
         1 = medulla
         2 = pons
         3 = mesencephalon

All augmentations apply the same geometric transform to both the image
(bilinear interpolation) and the mask (nearest-neighbour to preserve labels).

For 3D volumetric models, BrainstemVolumeDataset and BrainstemVolumeDataFactory
return full (B, 1, 80, 80, 96) volumes and (B, 80, 80, 96) masks.
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

# NIfTI label values and their names
NUM_CLASSES = 4  # 0=bg, 1=medulla, 2=pons, 3=mesencephalon
CLASS_NAMES = ["background", "medulla", "pons", "mesencephalon"]
FOREGROUND_LABELS = [1, 2, 3]

# ---------------------------------------------------------------------------
# Subject splits (seed=42)
# IDs available: 31..70 except 37 = 39 subjects
# ---------------------------------------------------------------------------
_ALL_IDS = [i for i in range(31, 71) if i != 37]


def get_subject_splits(seed: int = 42) -> Dict[str, List[int]]:
    """Return reproducible train/val/test subject ID splits (6/6/27)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(_ALL_IDS))
    ids = np.array(_ALL_IDS)[perm].tolist()
    return {
        "test":  sorted(ids[:6]),
        "val":   sorted(ids[6:12]),
        "train": sorted(ids[12:]),
    }


# ---------------------------------------------------------------------------
# Paired augmentation (image + mask)
# ---------------------------------------------------------------------------

class BrainstemAugmentation:
    """Paired tensor augmentation for 2D MRI slices.

    Applies the same geometric transform to the image (BILINEAR) and mask
    (NEAREST) so that label alignment is preserved.  Intensity perturbations
    are applied to the image only.

    Args:
        train: If False, this is a no-op (validation / test pass-through).
    """

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img:  float tensor (1, H, W) — z-score normalised MRI slice.
            mask: long tensor  (H, W)    — integer label map {0,1,2,3}.

        Returns:
            Augmented (img, mask) pair.
        """
        if not self.train:
            return img, mask

        # Convert mask to (1, H, W) float for torchvision ops, restore later
        mask_f = mask.unsqueeze(0).float()

        # ── Geometric ────────────────────────────────────────────────────────
        if random.random() < 0.5:
            img    = TF.hflip(img)
            mask_f = TF.hflip(mask_f)

        if random.random() < 0.25:
            img    = TF.vflip(img)
            mask_f = TF.vflip(mask_f)

        angle = random.uniform(-15.0, 15.0)
        img    = TF.rotate(img,    angle,
                           interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        mask_f = TF.rotate(mask_f, angle,
                           interpolation=TF.InterpolationMode.NEAREST,  fill=0.0)

        if random.random() < 0.5:
            tx    = random.uniform(-0.08, 0.08) * img.shape[-1]
            ty    = random.uniform(-0.08, 0.08) * img.shape[-2]
            scale = random.uniform(0.90, 1.10)
            img    = TF.affine(img,    angle=0, translate=[tx, ty], scale=scale,
                               shear=0, interpolation=TF.InterpolationMode.BILINEAR,
                               fill=0.0)
            mask_f = TF.affine(mask_f, angle=0, translate=[tx, ty], scale=scale,
                               shear=0, interpolation=TF.InterpolationMode.NEAREST,
                               fill=0.0)

        # ── Intensity (image only) ────────────────────────────────────────────
        if random.random() < 0.5:
            img = img + torch.randn_like(img) * 0.05

        if random.random() < 0.3:
            img = img * random.uniform(0.90, 1.10)

        if random.random() < 0.2:
            h, w  = img.shape[-2], img.shape[-1]
            bias  = torch.randn(1, 1, 2, 2) * 0.05
            bias  = F.interpolate(bias, size=(h, w), mode="bilinear", align_corners=False)
            img   = img + bias.squeeze(0)

        # Restore mask to (H, W) long
        mask = mask_f.squeeze(0).round().long()
        return img, mask


# ---------------------------------------------------------------------------
# 3D Volume augmentation
# ---------------------------------------------------------------------------

def _axial_affine(
    vol: torch.Tensor,
    seg: torch.Tensor,
    angle_deg: float,
    tx_n: float,
    ty_n: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a consistent 2D affine (axial H-W plane) to a 3D volume.

    Uses batched ``F.grid_sample`` over all depth slices — no Python loop.

    Args:
        vol:      float (1, H, W, D)
        seg:      long  (H, W, D)
        angle_deg: rotation angle in degrees
        tx_n, ty_n: normalised translation in [-1, 1] (positive → right/down)
        scale:    isotropic scale factor
    """
    C, H, W, D = vol.shape
    a = math.radians(angle_deg)
    ca, sa = math.cos(a) * scale, math.sin(a) * scale

    theta = torch.tensor(
        [[[ca, -sa, tx_n],
          [sa,  ca, ty_n]]],
        dtype=torch.float32,
    ).expand(D, -1, -1)  # (D, 2, 3)

    grid = F.affine_grid(theta, (D, 1, H, W), align_corners=False)  # (D, H, W, 2)

    # vol:  (1, H, W, D) → (D, 1, H, W)
    vol_in = vol.permute(3, 0, 1, 2)
    vol_out = F.grid_sample(vol_in, grid, mode="bilinear",  padding_mode="zeros", align_corners=False)
    vol_out = vol_out.permute(1, 2, 3, 0)  # (1, H, W, D)

    # seg:  (H, W, D) → (D, 1, H, W)
    seg_in  = seg.permute(2, 0, 1).unsqueeze(1).float()
    seg_out = F.grid_sample(seg_in, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    seg_out = seg_out.squeeze(1).permute(1, 2, 0).long()  # (H, W, D)

    return vol_out, seg_out


class BrainstemVolumeAugmentation:
    """Paired 3D augmentation matching the 2D pipeline.

    Geometric transforms act in the axial (H-W) plane, applied consistently
    across all depth slices via batched ``F.grid_sample``.
    Intensity perturbations are applied to the volume only.

    Args:
        train: If False, this is a no-op.
    """

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(
        self, vol: torch.Tensor, seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vol: float (1, H, W, D)
            seg: long  (H, W, D)
        """
        if not self.train:
            return vol, seg

        C, H, W, D = vol.shape

        # ── Rotation + optional affine (axial plane) ─────────────────────────
        angle_deg = random.uniform(-15.0, 15.0)
        tx_n, ty_n, scale = 0.0, 0.0, 1.0

        if random.random() < 0.5:
            tx_n  = random.uniform(-0.08, 0.08) * 2.0   # ±8 % → normalised
            ty_n  = random.uniform(-0.08, 0.08) * 2.0
            scale = random.uniform(0.90, 1.10)

        vol, seg = _axial_affine(vol, seg, angle_deg, tx_n, ty_n, scale)

        # ── Intensity (volume only) ───────────────────────────────────────────
        if random.random() < 0.5:
            vol = vol + torch.randn_like(vol) * 0.05

        if random.random() < 0.3:
            vol = vol * random.uniform(0.90, 1.10)

        if random.random() < 0.2:
            bias = torch.randn(1, 1, 2, 2, 2) * 0.05
            bias = F.interpolate(bias, size=(H, W, D), mode="trilinear", align_corners=False)
            vol  = vol + bias.squeeze(0)

        return vol, seg


# ---------------------------------------------------------------------------
# 2D Slice dataset
# ---------------------------------------------------------------------------

class BrainstemSliceDataset(Dataset):
    """2D axial slices extracted from 3D NIfTI brainstem volumes.

    For each subject, all axial slices with at least ``min_brainstem_voxels``
    foreground voxels are included.  Returns paired (image, mask) tensors.

    EVALUATION CAVEAT: this filter is applied to the val/test splits too, so the
    2D model is never scored on empty (non-brainstem) slices and incurs no
    false-positive penalty there. Reported 2D Dice is therefore OPTIMISTIC
    relative to full-volume inference, and is NOT directly comparable to the 3D
    models (which evaluate whole volumes via BrainstemVolumeDataset). For an
    honest 2D number, evaluate over all slices of held-out subjects (or at the
    volume level). This is disclosed in the paper's limitations.

    Args:
        data_dir: Path to the directory containing NIfTI files.
        subject_ids: List of integer subject IDs to load.
        augmentation: BrainstemAugmentation instance (or None for no aug).
        min_brainstem_voxels: Minimum non-zero label voxels to include a slice.
    """

    def __init__(
        self,
        data_dir: str,
        subject_ids: List[int],
        augmentation: Optional[BrainstemAugmentation] = None,
        min_brainstem_voxels: int = 20,
    ) -> None:
        self.data_dir  = Path(data_dir)
        self.augmentation = augmentation
        self.min_brainstem_voxels = min_brainstem_voxels

        # In-memory storage (small dataset: ~1200 slices × 80×80 × float32 ≈ 30 MB)
        self.images: List[np.ndarray] = []  # (80, 80) float32
        self.masks:  List[np.ndarray] = []  # (80, 80) int32,  values in {0,1,2,3}

        self._load_subjects(subject_ids)

        fg_voxels = sum(int((m > 0).sum()) for m in self.masks)
        logger.info(
            f"BrainstemSliceDataset: {len(self.masks)} slices from "
            f"{len(subject_ids)} subjects | "
            f"foreground voxels: {fg_voxels:,}"
        )

    def _load_subjects(self, subject_ids: List[int]) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "nibabel is required. Install with: pip install nibabel"
            ) from exc

        for sid in subject_ids:
            img_path = self.data_dir / f"a{sid}_n4bfc_mni_cropped_80x80x96.nii.gz"
            seg_path = self.data_dir / f"a{sid}-seg.nii.gz"

            if not img_path.exists():
                logger.warning(f"Image missing for subject {sid}: {img_path.name}")
                continue
            if not seg_path.exists():
                logger.warning(f"Segmentation missing for subject {sid}: {seg_path.name}")
                continue

            try:
                volume = nib.load(str(img_path)).get_fdata(dtype=np.float32)  # (80,80,96)
                seg    = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int32)
            except Exception as exc:
                logger.warning(f"Failed to load subject {sid}: {exc}")
                continue

            # Per-volume z-score normalisation
            mean, std = float(volume.mean()), float(volume.std()) + 1e-8
            volume = (volume - mean) / std

            # Iterate over axial slices (z = dim 2)
            for z in range(volume.shape[2]):
                seg_slice = seg[:, :, z]
                if int((seg_slice > 0).sum()) < self.min_brainstem_voxels:
                    continue
                self.images.append(volume[:, :, z].copy())
                self.masks.append(seg_slice.copy())

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img  = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, 80, 80)
        mask = torch.from_numpy(self.masks[idx]).long()          # (80, 80)

        if self.augmentation is not None:
            img, mask = self.augmentation(img, mask)

        return img, mask

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights (shape: [NUM_CLASSES]) for CE loss."""
        all_labels = np.concatenate([m.ravel() for m in self.masks])
        counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (NUM_CLASSES * counts)
        return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# 3D Volume dataset (for mipaim_unet / swin_unetr)
# ---------------------------------------------------------------------------

class BrainstemVolumeDataset(Dataset):
    """Full 3D NIfTI brainstem volumes for volumetric segmentation.

    Returns (img_volume, seg_volume) tensors:
        img_volume: float (1, 80, 80, 96) — z-score normalised
        seg_volume: long  (80, 80, 96)    — labels {0,1,2,3}
    """

    def __init__(
        self,
        data_dir: str,
        subject_ids: List[int],
        augmentation: Optional[BrainstemVolumeAugmentation] = None,
    ) -> None:
        self.augmentation = augmentation
        self.volumes: List[np.ndarray] = []   # (80, 80, 96) float32
        self.segs:    List[np.ndarray] = []   # (80, 80, 96) int32

        self._load_subjects(Path(data_dir), subject_ids)
        logger.info(
            f"BrainstemVolumeDataset: {len(self.volumes)} volumes "
            f"from {len(subject_ids)} subjects"
        )

    def _load_subjects(self, data_dir: Path, subject_ids: List[int]) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required. Install: pip install nibabel") from exc

        for sid in subject_ids:
            img_path = data_dir / f"a{sid}_n4bfc_mni_cropped_80x80x96.nii.gz"
            seg_path = data_dir / f"a{sid}-seg.nii.gz"
            if not img_path.exists() or not seg_path.exists():
                logger.warning(f"Subject {sid} missing, skipping.")
                continue
            try:
                vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
                seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int32)
            except Exception as exc:
                logger.warning(f"Failed to load subject {sid}: {exc}")
                continue
            mean, std = float(vol.mean()), float(vol.std()) + 1e-8
            self.volumes.append(((vol - mean) / std).copy())
            self.segs.append(seg.copy())

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol = torch.from_numpy(self.volumes[idx]).unsqueeze(0)  # (1, 80, 80, 96)
        seg = torch.from_numpy(self.segs[idx]).long()            # (80, 80, 96)

        if self.augmentation is not None:
            vol, seg = self.augmentation(vol, seg)

        return vol, seg

    def class_weights(self) -> torch.Tensor:
        all_labels = np.concatenate([s.ravel() for s in self.segs])
        counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (NUM_CLASSES * counts)
        return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# 2D Factory
# ---------------------------------------------------------------------------

def BrainstemDataFactory(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train/val/test 2D slice DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    ds_cfg   = cfg.dataset
    splits   = get_subject_splits(seed=cfg.get("seed", 42))
    min_vox  = ds_cfg.get("min_brainstem_voxels", 20)
    data_dir = ds_cfg.data_dir

    train_aug = BrainstemAugmentation(train=True)
    eval_aug  = BrainstemAugmentation(train=False)

    train_ds = BrainstemSliceDataset(data_dir, splits["train"], train_aug, min_vox)
    val_ds   = BrainstemSliceDataset(data_dir, splits["val"],   eval_aug,  min_vox)
    test_ds  = BrainstemSliceDataset(data_dir, splits["test"],  eval_aug,  min_vox)

    nw         = ds_cfg.get("num_workers", 0)
    batch_size = cfg.training.batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    logger.info(
        f"2D Splits → train: {len(train_ds)} slices ({len(splits['train'])} subjects) | "
        f"val: {len(val_ds)} ({len(splits['val'])}) | "
        f"test: {len(test_ds)} ({len(splits['test'])})"
    )
    return train_loader, val_loader, test_loader, train_ds.class_weights()


# ---------------------------------------------------------------------------
# 3D Factory
# ---------------------------------------------------------------------------

def BrainstemVolumeDataFactory(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train/val/test 3D volume DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    ds_cfg   = cfg.dataset
    splits   = get_subject_splits(seed=cfg.get("seed", 42))
    data_dir = ds_cfg.data_dir

    train_aug = BrainstemVolumeAugmentation(train=True)
    eval_aug  = BrainstemVolumeAugmentation(train=False)

    train_ds = BrainstemVolumeDataset(data_dir, splits["train"], train_aug)
    val_ds   = BrainstemVolumeDataset(data_dir, splits["val"],   eval_aug)
    test_ds  = BrainstemVolumeDataset(data_dir, splits["test"],  eval_aug)

    nw         = ds_cfg.get("num_workers", 0)
    batch_size = cfg.training.get("batch_size_3d", 1)  # volumes are large

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    logger.info(
        f"3D Splits → train: {len(train_ds)} vols | "
        f"val: {len(val_ds)} | test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, train_ds.class_weights()
