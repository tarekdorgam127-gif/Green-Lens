import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from greenlens.data.dataloader import create_dataloaders
from greenlens.models.architecture import build_resnet18


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GreenLens Improved Training (ResNet18)")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Learning rate for pretrained backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for FC classification head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--use-weighted-loss", action="store_true", default=True, help="Use smoothed class weights in CrossEntropyLoss")
    parser.add_argument("--use-scheduler", action="store_true", default=True, help="Use CosineAnnealingLR scheduler")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    return parser.parse_args()


def compute_class_weights(train_csv, idx_to_class, device):
    df = pd.read_csv(train_csv)
    counts_map = df["label"].value_counts().to_dict()
    counts = np.array([counts_map[idx_to_class[i]] for i in range(len(idx_to_class))], dtype=np.float32)

    # Smoothed inverse frequency (square root) to balance minority classes without destabilizing gradients
    weights = np.sqrt(counts.max() / counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_targets = []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    val_loss = running_loss / total
    val_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    return val_loss, val_acc, macro_f1, balanced_acc, all_preds, all_targets


def main():
    args = parse_args()

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("🚀 GreenLens Improved Training (Differential LR + Weighted Loss + Scheduler)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Config: Epochs={args.epochs}, Batch={args.batch_size}, Backbone-LR={args.backbone_lr}, Head-LR={args.head_lr}")
    print("=" * 75)

    # 2. Data Loaders
    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "validation.csv"

    print("\n📦 Loading datasets with enhanced augmentations...")
    train_loader, val_loader, _, class_to_idx, idx_to_class = create_dataloaders(
        train_csv=train_csv,
        validation_csv=val_csv,
        test_csv=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root_dir=ROOT,
    )

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"✅ Training samples:   {len(train_loader.dataset):,}")
    print(f"✅ Validation samples: {len(val_loader.dataset):,}")
    print(f"✅ Classes count:      {len(class_names)}")

    # 3. Model Architecture
    print("\n🧠 Initializing ResNet18 Architecture...")
    model = build_resnet18(num_classes=len(class_names), pretrained=True).to(device)

    # 4. Criterion with Class Weights
    if args.use_weighted_loss:
        class_weights = compute_class_weights(train_csv, idx_to_class, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("⚖️  Applied smoothed inverse-frequency class weights to CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. Optimizer with Differential Learning Rates
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    print(f"🎯 Optimizer: AdamW | Backbone LR={args.backbone_lr} | FC Head LR={args.head_lr} | Weight Decay={args.weight_decay}")

    # 6. Scheduler: CosineAnnealingLR
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        print("📈 LR Scheduler: CosineAnnealingLR enabled.")

    # 7. Training Loop
    history = []
    best_macro_f1 = -1.0
    best_preds = None
    best_targets = None
    best_epoch = -1
    best_model_path = MODEL_DIR / "best_model.pth"
    improved_path = MODEL_DIR / "improved_resnet18.pth"

    print("\n🏋️ Starting Improved Training...\n")
    header = f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Macro F1':^10} | {'Balanced Acc':^12} | {'Backbone LR':^11}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, macro_f1, balanced_acc, preds, targets = evaluate(
            model, val_loader, criterion, device
        )

        current_backbone_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "balanced_acc": round(balanced_acc, 4),
            "backbone_lr": current_backbone_lr,
        }
        history.append(epoch_stats)

        is_best = macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = macro_f1
            best_preds = preds
            best_targets = targets
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "macro_f1": macro_f1,
                "balanced_acc": balanced_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, best_model_path)
            torch.save(checkpoint, improved_path)

        flag = " 🌟 (Best)" if is_best else ""
        print(
            f"{epoch:^7} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc * 100:^8.2f}% | {macro_f1:^10.4f} | {balanced_acc * 100:^11.2f}% | {current_backbone_lr:^11.2e}{flag}"
        )

    # 8. Save Metrics
    metrics_file = REPORTS_DIR / "improved_training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n📊 Metrics saved to: {metrics_file}")
    print(f"💾 Best checkpoint saved to: {improved_path} (from Epoch {best_epoch})")

    # 9. Classification Report on Validation for Best Epoch
    print("\n" + "=" * 75)
    print("📋 Improved Detailed Per-Class Validation Report:")
    print("=" * 75)
    report_str = classification_report(
        best_targets,
        best_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_str)

    report_dict = classification_report(
        best_targets,
        best_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with open(REPORTS_DIR / "improved_classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)


if __name__ == "__main__":
    main()
