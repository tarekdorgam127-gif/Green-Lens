from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, class_to_idx, transform=None, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        raw_path = Path(row["path"])
        if not raw_path.is_absolute() and self.root_dir is not None:
            if (self.root_dir / raw_path).is_file():
                image_path = self.root_dir / raw_path
            elif (self.root_dir / "notebooks" / raw_path).is_file():
                image_path = (self.root_dir / "notebooks" / raw_path).resolve()
            else:
                image_path = (self.root_dir / raw_path).resolve()
        else:
            image_path = raw_path

        label = row["label"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[label]

        return image, label