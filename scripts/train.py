"""
train.py
--------
Full training loop for BirdSoundClassifier.

Usage:
    python train.py --processed_dir ./processed --epochs 50

Strategy:
  - Epochs 1–5:   backbone frozen, train head only (fast warm-up)
  - Epochs 6+:    full fine-tuning with lower LR for backbone
  - Mixed precision (AMP) for speed on RTX cards
  - Cosine annealing LR schedule
  - WeightedRandomSampler + weighted CrossEntropyLoss for imbalance
  - Best model saved by validation top-1 accuracy
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loaders, BirdSoundDataset
from model import build_model

# ── Helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: tuple = (1, 5)):
    """Compute top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def run_epoch(model, loader, criterion, optimizer, scaler, device,
              is_train: bool, epoch: int):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for specs, labels in loader:
            specs  = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(specs)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            total_loss += loss.item()
            top1_total += top1
            top5_total += top5
            n_batches  += 1

    return {
        "loss": total_loss / n_batches,
        "top1": top1_total / n_batches,
        "top5": top5_total / n_batches,
    }


# ── Training entry point ──────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_loader, val_loader, test_loader = make_loaders(
        processed_dir        = args.processed_dir,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
        use_weighted_sampler = True,
    )

    train_ds = BirdSoundDataset(args.processed_dir, split="train")
    num_classes   = train_ds.num_classes
    class_weights = train_ds.get_class_weights().to(device)

    # Model
    model = build_model(num_classes, device)
    print(f"Model parameters: {model.num_trainable_params:,}")

    # Loss — weighted CE + label smoothing
    criterion = nn.CrossEntropyLoss(
        weight         = class_weights,
        label_smoothing = 0.1,
    )

    # Optimizer — separate LR for backbone vs head
    head_params     = list(model.backbone.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters()
                       if p not in set(head_params)]

    optimizer = AdamW([
        {"params": head_params,     "lr": args.lr,          "weight_decay": 1e-4},
        {"params": backbone_params, "lr": args.lr * 0.1,    "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_top1 = 0.0
    history = []

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Warm-up (head only): epochs 1–{args.warmup_epochs}")
    print(f"  Full fine-tune:      epochs {args.warmup_epochs+1}–{args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Freeze/unfreeze backbone
        if epoch <= args.warmup_epochs:
            model.freeze_backbone()
        elif epoch == args.warmup_epochs + 1:
            model.unfreeze_all()
            print(f"  → Epoch {epoch}: unfreezing backbone, full fine-tune begins")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                  scaler, device, is_train=True, epoch=epoch)
        val_metrics   = run_epoch(model, val_loader, criterion, optimizer,
                                  scaler, device, is_train=False, epoch=epoch)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['top1']:.1f}% | "
            f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['top1']:.1f}% "
            f"top5 {val_metrics['top5']:.1f}% | "
            f"LR {lr_now:.2e} | {elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
            "lr": lr_now,
        })

        # Save best model
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_top1":    best_val_top1,
                "num_classes": num_classes,
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ New best: {best_val_top1:.2f}% (saved)")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict(),
                "val_top1":        val_metrics["top1"],
                "num_classes":     num_classes,
            }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = run_epoch(model, test_loader, criterion, optimizer,
                             scaler, device, is_train=False, epoch=0)
    print(f"\nTest  → loss {test_metrics['loss']:.4f} | "
          f"top-1 {test_metrics['top1']:.2f}% | top-5 {test_metrics['top5']:.2f}%")
    print(f"Best val top-1: {best_val_top1:.2f}%")
    print(f"\nCheckpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",   type=str, default="./processed")
    parser.add_argument("--checkpoint_dir",  type=str, default="./checkpoints")
    parser.add_argument("--epochs",          type=int, default=50)
    parser.add_argument("--warmup_epochs",   type=int, default=5,
                        help="Epochs to train head only before unfreezing backbone")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--lr",              type=float, default=3e-4,
                        help="LR for classifier head; backbone gets lr*0.1")
    parser.add_argument("--num_workers",     type=int, default=4)
    args = parser.parse_args()

    train(args)
