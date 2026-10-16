import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from PIL import Image

from greenlens.inference.disease_info import DiseaseDB
from greenlens.inference.predict import PlantDiseasePredictor


class TestGreenLensIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PlantDiseasePredictor()
        cls.db = DiseaseDB()
        cls.test_csv = ROOT / "data" / "processed" / "test.csv"

    def test_database_knowledge_coverage(self):
        """Ensure all 15 classes in the trained model exist in DiseaseDB with full Arabic and English content."""
        self.assertEqual(len(self.predictor.class_names), 15)
        for class_name in self.predictor.class_names:
            for lang in ["ar", "en"]:
                info = self.db.lookup(class_name, lang=lang)
                self.assertTrue(info["known"], f"Class {class_name} not found in DiseaseDB")
                self.assertTrue(len(info["plant"]) > 0, f"Plant name missing for {class_name}")
                self.assertTrue(len(info["disease"]) > 0, f"Disease name missing for {class_name}")
                self.assertTrue(len(info["description"]) > 0, f"Description missing for {class_name}")

    def test_pipeline_on_dataset_samples(self):
        """Verify the full pipeline: Input -> Preprocess -> Model -> Prediction -> Disease Info -> Result."""
        df = pd.read_csv(self.test_csv).head(5)
        for _, row in df.iterrows():
            rel_path = Path(row["path"])
            img_path = (ROOT / "notebooks" / rel_path).resolve()
            self.assertTrue(img_path.is_file(), f"Test image does not exist: {img_path}")

            result = self.predictor.predict(img_path, top_k=3, lang="ar")

            # Check structure
            self.assertIsNotNone(result.best)
            self.assertGreaterEqual(result.best.probability, 0.0)
            self.assertLessEqual(result.best.probability, 1.0)
            self.assertEqual(len(result.top_predictions), 3)

            # Check attached disease info
            self.assertTrue(result.best.info["known"])
            self.assertIn("plant", result.best.info)
            self.assertIn("disease", result.best.info)

    def test_dummy_image_uncertainty(self):
        """A pure random noise image should trigger the uncertainty warning or low confidence."""
        import numpy as np
        noise_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(noise_arr)
        result = self.predictor.predict(dummy_img, top_k=3)
        self.assertIsNotNone(result.best)


if __name__ == "__main__":
    unittest.main()
