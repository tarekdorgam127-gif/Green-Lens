import pandas as pd

from torch.utils.data import DataLoader

from greenlens.data.dataset import PlantDiseaseDataset
from greenlens.data.preprocessing import (
    get_train_transforms,
    get_eval_transforms,
)


def get_class_mapping(csv_file):
    df = pd.read_csv(csv_file)

    classes = sorted(df["label"].unique())

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    return class_to_idx, idx_to_class


def create_dataloaders(
    train_csv,
    validation_csv,
    test_csv=None,
    batch_size=32,
    num_workers=4,
    root_dir=None,
):
    class_to_idx, idx_to_class = get_class_mapping(train_csv)

    train_dataset = PlantDiseaseDataset(
        train_csv,
        class_to_idx,
        transform=get_train_transforms(),
        root_dir=root_dir,
    )

    validation_dataset = PlantDiseaseDataset(
        validation_csv,
        class_to_idx,
        transform=get_eval_transforms(),
        root_dir=root_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_csv is not None:
        test_dataset = PlantDiseaseDataset(
            test_csv,
            class_to_idx,
            transform=get_eval_transforms(),
            root_dir=root_dir,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class,
    )