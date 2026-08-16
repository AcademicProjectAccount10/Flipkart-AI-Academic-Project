"""Part 2, Tasks 1--2: Fashion-MNIST loading, splitting, and preprocessing only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
VALIDATION_SIZE = 5_000
BACKBONE = "ResNet-18"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FashionMNISTData:
    """Reusable Part 2 datasets and DataLoaders; no model is created here."""

    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def build_preprocessing_transform() -> transforms.Compose:
    """Convert Fashion-MNIST grayscale images for an ImageNet-pretrained ResNet-18."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_datasets_and_loaders(
    batch_size: int = 64, num_workers: int = 0
) -> FashionMNISTData:
    """Download Fashion-MNIST, make a stratified validation split, and return loaders.

    The official test split is instantiated independently and is never included in
    the split operation, training loader, or validation loader.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    transform = build_preprocessing_transform()

    # This untransformed instance provides the 60,000 official training labels for
    # stratification only. download=True is deliberately retained for reproducibility.
    split_source = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True)
    labels = np.asarray(split_source.targets)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    train_indices, validation_indices = next(splitter.split(np.zeros(len(labels)), labels))

    # Separate transformed dataset views use only official training indices.
    transformed_training_source = datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    transformed_validation_source = datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    train_dataset = Subset(transformed_training_source, train_indices.tolist())
    validation_dataset = Subset(transformed_validation_source, validation_indices.tolist())

    # The official 10,000-image test split remains completely untouched.
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    if (len(train_dataset), len(validation_dataset), len(test_dataset)) != (55_000, 5_000, 10_000):
        raise ValueError("Unexpected Fashion-MNIST split sizes.")

    return FashionMNISTData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        validation_loader=DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def write_setup_report(data: FashionMNISTData) -> Path:
    """Write the Task 1--2 setup report without evaluating a model."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / "part2_tasks_1_2_setup_report.md"
    report_path.write_text(
        f"""# Part 2 — Tasks 1–2 Setup

## Dataset source

The dataset is the official Fashion-MNIST benchmark from Zalando Research, loaded exclusively with `torchvision.datasets.FashionMNIST(root=data/raw, download=True)`. It contains the official 60,000-image training split and 10,000-image test split.

## Stratified split sizes

| Split | Images | Source |
|---|---:|---|
| Training | {len(data.train_dataset):,} | Official training split |
| Validation | {len(data.validation_dataset):,} | Stratified split from official training data |
| Test | {len(data.test_dataset):,} | Official untouched test split |

The validation split uses `StratifiedShuffleSplit(test_size=5000, random_state=42)` over only the official training labels. The test set is created separately and is not used for training or validation.

## Preprocessing for a pretrained backbone

Chosen backbone: **{BACKBONE}**. Required input image size: **{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}** pixels.

The preprocessing pipeline is: replicate the original one-channel grayscale image to three channels with `Grayscale(num_output_channels=3)`, resize to {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}, convert to a tensor, then normalize with ImageNet mean `{IMAGENET_MEAN}` and standard deviation `{IMAGENET_STD}`. This script creates reusable PyTorch `Dataset` and `DataLoader` objects only; it does not construct, train, fine-tune, evaluate, save, or export a model.
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    data = create_datasets_and_loaders()
    report_path = write_setup_report(data)
    sample_images, sample_labels = next(iter(data.train_loader))
    print(f"Backbone: {BACKBONE}; image size: {IMAGE_SIZE}")
    print(
        f"Split sizes — train: {len(data.train_dataset)}, validation: {len(data.validation_dataset)}, "
        f"test: {len(data.test_dataset)}"
    )
    print(f"One preprocessed training batch: images={tuple(sample_images.shape)}, labels={tuple(sample_labels.shape)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
