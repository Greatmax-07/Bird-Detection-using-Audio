"""
evaluate_tta.py
---------------
Evaluates the best model with test-time augmentation (TTA).
For each test chunk, runs N augmented versions and averages softmax outputs.

Usage:
    python scripts/evaluate_tta.py \
        --checkpoint checkpoints_v4/best_model.pt \
        --processed_dir dataset/processed \
        --n_tta 5
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from dataset import BirdSoundDataset
from model import BirdSoundClassifier


def augment_spec(spec: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Apply a random mild augmentation to a single spectrogram (3, 128, 216)."""
    s = spec.clone()
    F_bins, T = s.shape[1], s.shape[2]

    # random freq mask
    f = random.randint(0, 20)
    f0 = random.randint(0, max(0, F_bins - f))
    s[:, f0:f0+f, :] = 0.0

    # random time mask
    t = random.randint(0, 30)
    t0 = random.randint(0, max(0, T - t))
    s[:, :, t0:t0+t] = 0.0

    # random time shift (roll)
    shift = random.randint(-20, 20)
    s = torch.roll(s, shift, dims=2)

    return s


def evaluate_tta(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds = BirdSoundDataset(args.processed_dir, split="test", augment=False)
    with open(Path(args.processed_dir) / "stats.json") as f:
        stats = json.load(f)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = BirdSoundClassifier(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    correct_top1 = correct_top5 = total = 0

    for i in tqdm(range(len(ds)), desc="TTA eval"):
        spec, label = ds[i]  # (3, 128, 216), int

        # Build N augmented versions (first is clean)
        versions = [spec]
        for _ in range(args.n_tta - 1):
            versions.append(augment_spec(spec, stats["mean"], stats["std"]))

        batch = torch.stack(versions).to(device)  # (N, 3, 128, 216)

        with torch.no_grad(), autocast('cuda'):
            logits = model(batch)               # (N, num_classes)
            probs  = F.softmax(logits, dim=1)   # (N, num_classes)
            avg_probs = probs.mean(dim=0)        # (num_classes,)

        top5 = avg_probs.topk(5).indices.tolist()
        if top5[0] == label:
            correct_top1 += 1
        if label in top5:
            correct_top5 += 1
        total += 1

    top1 = correct_top1 / total * 100
    top5 = correct_top5 / total * 100
    print(f"\nTTA (n={args.n_tta})  →  top-1: {top1:.2f}%  |  top-5: {top5:.2f}%  |  n={total}")
    return top1, top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--processed_dir",  type=str, default="dataset/processed")
    parser.add_argument("--n_tta",          type=int, default=5,
                        help="Number of augmented versions to average (1 = no TTA)")
    args = parser.parse_args()
    evaluate_tta(args)
