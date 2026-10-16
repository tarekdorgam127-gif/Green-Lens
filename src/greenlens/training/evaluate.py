import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from greenlens.data.dataset import PlantDiseaseDataset
from greenlens.data.preprocessing import get_eval_transforms
from greenlens.models.architecture import build_resnet18


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "best_model.pth"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_test_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("🎯 GreenLens Final Test Set Evaluation (Unseen Held-Out Data)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print("=" * 75)

    # 1. Load Checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint.get("idx_to_class", {idx: name for name, idx in class_to_idx.items()})
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 2. Build Model
    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Create Test DataLoader
    test_csv = DATA_DIR / "test.csv"
    test_dataset = PlantDiseaseDataset(
        csv_file=test_csv,
        class_to_idx=class_to_idx,
        transform=get_eval_transforms(),
        root_dir=ROOT,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"✅ Loaded {len(test_dataset):,} unseen test images across {len(class_names)} classes.")

    # 4. Evaluation Loop
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    print("\n⏳ Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # 5. Compute Metrics
    test_loss = running_loss / total
    test_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    print("\n" + "=" * 75)
    print("🏆 FINAL TEST SET PERFORMANCE (UNBIASED BENCHMARK)")
    print("=" * 75)
    print(f"  • Test Loss:              {test_loss:.4f}")
    print(f"  • Test Overall Accuracy:  {test_acc * 100:.2f}%")
    print(f"  • Test Macro F1-Score:    {macro_f1:.4f}")
    print(f"  • Test Weighted F1-Score: {weighted_f1:.4f}")
    print(f"  • Test Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"  • Misclassified Images:   {total - int(test_acc * total)} / {total}")
    print("=" * 75)

    # 6. Detailed Per-Class Report
    print("\n📋 Detailed Per-Class Performance on Test Set:")
    print("-" * 75)
    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_str)

    # 7. Save Results
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    test_metrics = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_weighted_f1": round(weighted_f1, 4),
        "test_balanced_accuracy": round(balanced_acc, 4),
        "total_test_samples": total,
        "correct_predictions": int(test_acc * total),
        "misclassified": total - int(test_acc * total),
        "classification_report": report_dict,
    }

    out_file = REPORTS_DIR / "final_test_metrics.json"
    with open(out_file, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"💾 Full test results saved to: {out_file}\n")


if __name__ == "__main__":
    evaluate_test_set()
