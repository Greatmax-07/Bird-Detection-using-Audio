import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loaders, BirdSoundDataset
from model import build_model


def accuracy(logits, labels, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size).item() for k in topk]


def mixup_batch(specs, labels, num_classes, alpha=0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(specs.size(0), device=specs.device)
    mixed = lam * specs + (1 - lam) * specs[idx]
    # soft labels
    y1 = torch.zeros(specs.size(0), num_classes, device=specs.device).scatter_(1, labels.unsqueeze(1), 1)
    y2 = y1[idx]
    return mixed, lam * y1 + (1 - lam) * y2


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train, num_classes, use_mixup):
    model.train() if is_train else model.eval()
    total_loss = top1_total = top5_total = n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for specs, labels in loader:
            specs  = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                if is_train and use_mixup:
                    mixed, soft_labels = mixup_batch(specs, labels, num_classes)
                    logits = model(mixed)
                    loss = (-soft_labels * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                else:
                    logits = model(specs)
                    loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            top1, top5 = accuracy(logits, labels)
            total_loss += loss.item()
            top1_total += top1
            top5_total += top5
            n_batches  += 1

    return {"loss": total_loss/n_batches, "top1": top1_total/n_batches, "top5": top5_total/n_batches}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = make_loaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )

    train_ds      = BirdSoundDataset(args.processed_dir, split="train")
    num_classes   = train_ds.num_classes
    class_weights = train_ds.get_class_weights().to(device)

    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    head_params     = list(model.backbone.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p not in set(head_params)]
    optimizer = AdamW([
        {"params": head_params,     "lr": args.lr,       "weight_decay": 1e-4},
        {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler('cuda')

    ckpt_dir      = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val_top1 = ckpt.get("val_top1", 0.0)
        # restore optimizer/scheduler only if full checkpoint (not best_model.pt)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
        model.unfreeze_all()
        print(f"Resumed from {args.resume}  (prev best val: {best_val_top1:.2f}%)")

    print(f"\nTraining epochs {start_epoch}–{start_epoch + args.epochs - 1}  |  mixup={'on' if args.mixup else 'off'}\n")

    history = []
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        if not args.resume or epoch > args.warmup_epochs:
            model.unfreeze_all()
        if epoch <= args.warmup_epochs and not args.resume:
            model.freeze_backbone()

        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler,
                            device, is_train=True,  num_classes=num_classes, use_mixup=args.mixup)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, scaler,
                            device, is_train=False, num_classes=num_classes, use_mixup=False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"Train loss {train_m['loss']:.4f} acc {train_m['top1']:.1f}% | "
            f"Val loss {val_m['loss']:.4f} acc {val_m['top1']:.1f}% top5 {val_m['top5']:.1f}% | "
            f"LR {lr_now:.2e} | {time.time()-t0:.0f}s"
        )

        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"val_{k}":   v for k, v in val_m.items()},
                        "lr": lr_now})

        if val_m["top1"] > best_val_top1:
            best_val_top1 = val_m["top1"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_top1": best_val_top1, "num_classes": num_classes},
                       ckpt_dir / "best_model.pt")
            print(f"  ✓ New best: {best_val_top1:.2f}% (saved)")

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "val_top1": val_m["top1"], "num_classes": num_classes},
                       ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = run_epoch(model, test_loader, criterion, optimizer, scaler,
                       device, is_train=False, num_classes=num_classes, use_mixup=False)
    print(f"Test → loss {test_m['loss']:.4f} | top-1 {test_m['top1']:.2f}% | top-5 {test_m['top5']:.2f}%")
    print(f"Best val top-1: {best_val_top1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",  type=str, default="./processed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best_model.pt)")
    parser.add_argument("--epochs",         type=int, default=30)
    parser.add_argument("--warmup_epochs",  type=int, default=5)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--mixup",          action="store_true",
                        help="Enable mixup augmentation during training")
    args = parser.parse_args()
    train(args)
