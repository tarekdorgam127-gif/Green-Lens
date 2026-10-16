from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image, ImageOps

from greenlens.data.preprocessing import get_eval_transforms
from greenlens.inference.disease_info import DiseaseDB
from greenlens.models.architecture import build_resnet18


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pth"


@dataclass
class ClassPrediction:
    class_name: str
    probability: float
    info: dict


@dataclass
class PredictionResult:
    top_predictions: List[ClassPrediction]
    best: ClassPrediction
    is_uncertain: bool
    confidence_threshold: float

    @property
    def is_healthy(self) -> bool:
        return self.best.info.get("healthy", False)


class PlantDiseasePredictor:
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        min_confidence: float = 0.60,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_confidence = min_confidence
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_MODEL_PATH)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        # Load Checkpoint and Class Metadata
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = checkpoint.get(
            "idx_to_class", {idx: name for name, idx in self.class_to_idx.items()}
        )
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # Build Model and Load Weights
        self.model = build_resnet18(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transformation Pipeline & Knowledge Base
        self.transform = get_eval_transforms()
        self.db = DiseaseDB()

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 3,
        lang: str = "ar",
    ) -> PredictionResult:
        """Run end-to-end inference on a single image and return detailed predictions."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input

        # Handle EXIF orientation and ensure RGB
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward Pass
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

        # Top-K
        k = min(top_k, len(self.class_names))
        top_probs, top_indices = probs.topk(k)

        top_preds: List[ClassPrediction] = []
        for prob_val, idx_val in zip(top_probs, top_indices):
            c_name = self.class_names[idx_val.item()]
            c_prob = float(prob_val.item())
            info = self.db.lookup(c_name, lang=lang)
            top_preds.append(ClassPrediction(class_name=c_name, probability=c_prob, info=info))

        best = top_preds[0]
        is_uncertain = best.probability < self.min_confidence

        return PredictionResult(
            top_predictions=top_preds,
            best=best,
            is_uncertain=is_uncertain,
            confidence_threshold=self.min_confidence,
        )


def main():
    parser = argparse.ArgumentParser(description="GreenLens Single Image Prediction")
    parser.add_argument("image_path", type=str, help="Path to input plant leaf image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--lang", type=str, default="ar", choices=["ar", "en"], help="Display language")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    predictor = PlantDiseasePredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.image_path, top_k=args.top_k, lang=args.lang)

    print("=" * 65)
    print("🌿 GreenLens Diagnostic Result:")
    print("=" * 65)
    best_info = result.best.info
    print(f"Plant:      {best_info['plant']}")
    print(f"Diagnosis:  {best_info['disease']}")
    print(f"Confidence: {result.best.probability * 100:.2f}%")
    severity_label = best_info.get("severity_label", "-")
    status_text = "Healthy ✅" if result.is_healthy else f"Disease Detected ⚠️ (Severity: {severity_label})"
    print(f"Status:     {status_text}")

    if result.is_uncertain:
        print(f"\n⚠️  Warning: Low confidence (< {result.confidence_threshold * 100:.0f}%). Photo might be ambiguous or out-of-domain.")

    print("\n📊 Top Predictions:")
    for i, pred in enumerate(result.top_predictions, 1):
        print(f"  {i}. {pred.info['plant']} - {pred.info['disease']}: {pred.probability * 100:.2f}%")

    if not result.is_healthy and best_info.get("treatment"):
        print("\n💊 Recommended Treatment:")
        for t in best_info["treatment"]:
            print(f"  • {t}")

    if not result.is_healthy and best_info.get("prevention"):
        print("\n🛡️ Prevention Tips:")
        for p in best_info["prevention"]:
            print(f"  • {p}")


if __name__ == "__main__":
    main()
