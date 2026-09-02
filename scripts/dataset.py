"""
dataset.py
----------
PyTorch Dataset for preprocessed mel spectrogram chunks.
Handles normalization, SpecAugment, and class weight computation.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# ── SpecAugment ─────────────────────────────────────────────────────────────────

class SpecAugment:
    """
    Randomly masks contiguous frequency bands and time steps on a mel spectrogram.
    Applied only during training.

    Args:
        freq_mask_max:  max number of mel bins to mask per band
        time_mask_max:  max number of time frames to mask per stripe
        n_freq_masks:   number of frequency masks
        n_time_masks:   number of time masks
    """
    def __init__(self,
                 freq_mask_max: int = 27,
                 time_mask_max: int = 40,
                 n_freq_masks: int  = 2,
                 n_time_masks: int  = 2):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.n_freq_masks  = n_freq_masks
        self.n_time_masks  = n_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: Tensor of shape (C, F, T) — channels, freq bins, time frames
        Returns augmented tensor of the same shape.
        """
        spec = spec.clone()
        _, F, T = spec.shape
        fill_value = spec.mean()

        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_max)
            f0 = random.randint(0, max(0, F - f))
            spec[:, f0 : f0 + f, :] = fill_value

        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_max)
            t0 = random.randint(0, max(0, T - t))
            spec[:, :, t0 : t0 + t] = fill_value

        return spec


# ── Dataset ──────────────────────────────────────────────────────────────────────

class BirdSoundDataset(Dataset):
    """
    Loads preprocessed mel spectrogram .npy files.

    Args:
        processed_dir:  root of the processed/ directory
        split:          'train', 'val', or 'test'
        augment:        whether to apply SpecAugment (train only)
        target_length:  fixed number of time frames (pads/crops). None = no resize.
    """

    TARGET_TIME_FRAMES = 216  # ≈ 5s at sr=22050, hop=512

    def __init__(self,
                 processed_dir: str | Path,
                 split: str = "train",
                 augment: bool = False,
                 target_length: int | None = TARGET_TIME_FRAMES):

        self.root      = Path(processed_dir)
        self.split     = split
        self.augment   = augment
        self.target_length = target_length

        # Load metadata
        df = pd.read_csv(self.root / "metadata.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)

        with open(self.root / "class_map.json") as f:
            self.class_map = json.load(f)
        self.num_classes = len(self.class_map)

        with open(self.root / "stats.json") as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std  = stats["std"]

        # Augmentation (only for training)
        # self.spec_augment = SpecAugment() if augment else None
        self.spec_augment = SpecAugment(
            freq_mask_max=40,   # was 27
            time_mask_max=60,   # was 40
            n_freq_masks=2,
            n_time_masks=2
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        spec  = np.load(self.root / row["path"])           # (128, T)
        label = int(row["class_idx"])

        # Standardize time axis
        if self.target_length is not None:
            spec = self._fix_length(spec, self.target_length)

        # Normalize
        spec = (spec - self.mean) / (self.std + 1e-6)

        # (128, T) → (1, 128, T) → (3, 128, T)  [repeat for ImageNet compat]
        spec_t = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1).float()

        # SpecAugment
        if self.spec_augment is not None:
            spec_t = self.spec_augment(spec_t)

        return spec_t, label

    @staticmethod
    def _fix_length(spec: np.ndarray, target: int) -> np.ndarray:
        """Crop or zero-pad the time axis to exactly `target` frames."""
        T = spec.shape[1]
        if T >= target:
            return spec[:, :target]
        pad = np.zeros((spec.shape[0], target - T), dtype=np.float32)
        return np.concatenate([spec, pad], axis=1)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for WeightedRandomSampler
        or weighted CrossEntropyLoss.
        Returns a tensor of shape (num_classes,).
        """
        counts = self.df["class_idx"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights = weights / weights.sum() * len(weights)   # normalize so mean ≈ 1
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor(
            [class_weights[idx].item() for idx in self.df["class_idx"]],
            dtype=torch.float32
        )
        return sample_weights


# ── DataLoader factory ────────────────────────────────────────────────────────────

def make_loaders(
    processed_dir: str | Path,
    batch_size:    int = 32,
    num_workers:   int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    use_weighted_sampler: if True, oversamples rare classes during training.
    Combine with class weights in the loss for best results.
    """
    train_ds = BirdSoundDataset(processed_dir, split="train", augment=True)
    val_ds   = BirdSoundDataset(processed_dir, split="val",   augment=False)
    test_ds  = BirdSoundDataset(processed_dir, split="test",  augment=False)

    if use_weighted_sampler and len(train_ds) > 0:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(train_ds),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"Train: {len(train_ds):,} chunks | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"Classes: {train_ds.num_classes}")

    return train_loader, val_loader, test_loader
