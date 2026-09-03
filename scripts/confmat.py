"""
confmat.py
----------
Prints per-class accuracy and top confusion pairs for the test set.

Usage:
    python scripts/confmat.py \
        --checkpoint checkpoints_v4/best_model.pt \
        --processed_dir dataset/processed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from dataset import BirdSoundDataset
from model import BirdSoundClassifier


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BirdSoundDataset(args.processed_dir, split="test", augment=False)
    with open(Path(args.processed_dir) / "class_map.json") as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}
    n = len(class_map)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = BirdSoundClassifier(num_classes=n)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    confmat = np.zeros((n, n), dtype=int)

    for i in tqdm(range(len(ds)), desc="Evaluating"):
        spec, label = ds[i]
        with torch.no_grad(), autocast('cuda'):
            logits = model(spec.unsqueeze(0).to(device))
            pred   = logits.argmax(dim=1).item()
        confmat[label, pred] += 1

    # Per-class accuracy
    print("\n── Per-class accuracy (test set) ──────────────────────────────")
    per_class = []
    for i in range(n):
        total   = confmat[i].sum()
        correct = confmat[i, i]
        acc     = correct / total * 100 if total > 0 else 0.0
        per_class.append((acc, idx_to_class[i], correct, total))
    per_class.sort()
    for acc, name, correct, total in per_class:
        bar  = "█" * int(acc // 5)
        flag = " ←" if acc < 40 else ""
        print(f"  {name:45s} {acc:5.1f}%  ({correct}/{total})  {bar}{flag}")

    # Top confused pairs
    print("\n── Most confused pairs ────────────────────────────────────────")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and confmat[i, j] > 0:
                pairs.append((confmat[i, j], idx_to_class[i], idx_to_class[j]))
    pairs.sort(reverse=True)
    for count, true_cls, pred_cls in pairs[:20]:
        print(f"  {true_cls:45s} → predicted as {pred_cls:45s}  ({count}x)")

    overall = np.diag(confmat).sum() / confmat.sum() * 100
    print(f"\nOverall test top-1: {overall:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default="dataset/processed")
    args = parser.parse_args()
    evaluate(args)
