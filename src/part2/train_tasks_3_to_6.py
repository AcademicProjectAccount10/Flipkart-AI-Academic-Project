"""Part 2, Tasks 3--6: cached ResNet-18 feature extraction and evaluation only.

This script intentionally does not save model weights or export images (Tasks 7--8).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from setup_fashion_mnist import IMAGE_SIZE, create_datasets_and_loaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
BATCH_SIZE = 128
FEATURE_EXTRACTION_LEARNING_RATE = 1e-3
FEATURE_EXTRACTION_EPOCHS = 12
FINE_TUNE_LEARNING_RATE = 1e-4
FINE_TUNE_EPOCHS = 2


class FashionResNet18(nn.Module):
    """Pretrained ResNet-18 feature extractor plus a new 10-class head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(in_features, len(CLASS_NAMES))

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.backbone(images))


@dataclass
class CachedFeatures:
    """One-pass frozen-backbone feature vectors and labels."""

    features: Tensor
    labels: Tensor


def set_seed() -> None:
    """Make the split-independent training stages reproducible."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def cache_frozen_features(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> CachedFeatures:
    """Run the frozen backbone exactly once over a loader and retain CPU features."""
    backbone.eval()
    feature_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            features = backbone(images.to(device)).cpu()
            feature_batches.append(features)
            label_batches.append(labels.cpu())
    return CachedFeatures(torch.cat(feature_batches), torch.cat(label_batches))


def train_classifier_head(
    classifier: nn.Module, train_cache: CachedFeatures, device: torch.device
) -> tuple[list[dict[str, float]], nn.Module]:
    """Train only the replacement classifier head over cached frozen features."""
    cached_loader = DataLoader(
        TensorDataset(train_cache.features, train_cache.labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    classifier.to(device)
    classifier.train()
    optimizer = Adam(classifier.parameters(), lr=FEATURE_EXTRACTION_LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, FEATURE_EXTRACTION_EPOCHS + 1):
        total_loss = 0.0
        total_examples = 0
        for features, labels in cached_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
        history.append(
            {
                "stage": "feature_extraction_head_only",
                "epoch": epoch,
                "learning_rate": FEATURE_EXTRACTION_LEARNING_RATE,
                "mean_training_loss": total_loss / total_examples,
            }
        )
    return history, classifier


def cached_accuracy(classifier: nn.Module, cache: CachedFeatures, device: torch.device) -> float:
    """Evaluate the replacement head against cached backbone outputs."""
    classifier.eval()
    with torch.inference_mode():
        predictions = classifier(cache.features.to(device)).argmax(dim=1).cpu()
    return accuracy_score(cache.labels.numpy(), predictions.numpy())


def fine_tune_late_layer(
    model: FashionResNet18, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device
) -> tuple[list[dict[str, float]], float]:
    """Unfreeze layer4 only, retaining frozen early and middle ResNet layers."""
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = True
    model.to(device)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=FINE_TUNE_LEARNING_RATE,
    )
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
        validation_accuracy, _, _ = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "stage": "fine_tune_layer4_and_head",
                "epoch": epoch,
                "learning_rate": FINE_TUNE_LEARNING_RATE,
                "mean_training_loss": total_loss / total_examples,
                "validation_accuracy": validation_accuracy,
            }
        )
    return history, validation_accuracy


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate after selection; returns real labels and predictions."""
    model.eval()
    labels_all: list[Tensor] = []
    predictions_all: list[Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels_all.append(labels.cpu())
            predictions_all.append(predictions)
    labels_array = torch.cat(labels_all).numpy()
    predictions_array = torch.cat(predictions_all).numpy()
    return accuracy_score(labels_array, predictions_array), labels_array, predictions_array


def confusion_pair_explanation(actual: str, predicted: str) -> str:
    """Explain the observed pair with category-specific visual reasoning."""
    upper_garments = {"T-shirt/top", "Pullover", "Dress", "Coat", "Shirt"}
    footwear = {"Sandal", "Sneaker", "Ankle boot"}
    if actual in upper_garments and predicted in upper_garments:
        return (
            f"{actual} and {predicted} are both grayscale upper-body garment silhouettes. "
            "At 28×28 resolution, sleeves, collars, and fabric texture can be too faint to separate them reliably."
        )
    if actual in footwear and predicted in footwear:
        return (
            f"{actual} and {predicted} are both footwear side-profile silhouettes. "
            "Their shared sole and toe outlines can obscure the smaller shape cues that distinguish them in low-resolution grayscale images."
        )
    return (
        f"{actual} and {predicted} can have overlapping coarse silhouettes after resizing from Fashion-MNIST's 28×28 grayscale images. "
        "The limited resolution and absent colour/texture information make this observed mistake visually plausible."
    )


def main() -> None:
    set_seed()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = create_datasets_and_loaders(batch_size=BATCH_SIZE)
    model = FashionResNet18().to(device)

    # Caching happens once per train/validation dataset while the full backbone is frozen.
    train_cache = cache_frozen_features(model.backbone, data.train_loader, device)
    validation_cache = cache_frozen_features(model.backbone, data.validation_loader, device)
    history, trained_head = train_classifier_head(model.classifier, train_cache, device)
    model.classifier = trained_head
    validation_before = cached_accuracy(model.classifier, validation_cache, device)

    fine_tuning_required = validation_before < 0.80
    validation_after = validation_before
    if fine_tuning_required:
        fine_tune_history, validation_after = fine_tune_late_layer(
            model, data.train_loader, data.validation_loader, device
        )
        history.extend(fine_tune_history)

    # Test evaluation is intentionally deferred until the validation-driven model choice is complete.
    test_accuracy, test_labels, test_predictions = evaluate_model(model, data.test_loader, device)
    matrix = confusion_matrix(test_labels, test_predictions, labels=list(range(len(CLASS_NAMES))))
    class_report = pd.DataFrame(
        classification_report(
            test_labels,
            test_predictions,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    top_pair_indices = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[::-1], off_diagonal.shape))[0]
    pair_rows: list[dict[str, object]] = []
    for actual_index, predicted_index in top_pair_indices:
        if actual_index == predicted_index:
            continue
        actual, predicted = CLASS_NAMES[actual_index], CLASS_NAMES[predicted_index]
        pair_rows.append(
            {
                "actual_class": actual,
                "predicted_class": predicted,
                "misclassified_images": int(matrix[actual_index, predicted_index]),
                "explanation": confusion_pair_explanation(actual, predicted),
            }
        )
        if len(pair_rows) == 2:
            break
    confusion_pairs = pd.DataFrame(pair_rows)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "part2_tasks_3_6_training_history.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "part2_tasks_3_6_confusion_matrix.csv"
    )
    class_report.to_csv(OUTPUT_DIR / "part2_tasks_3_6_classification_report.csv")
    confusion_pairs.to_csv(OUTPUT_DIR / "part2_tasks_3_6_top_confusion_pairs.csv", index=False)

    fine_tuning_statement = (
        f"Fine-tuning was required because feature extraction reached {validation_before:.4f}, below 0.8000. "
        f"Only ResNet-18 layer4 and the classifier head were unfrozen, using Adam at {FINE_TUNE_LEARNING_RATE}; "
        f"validation accuracy then reached {validation_after:.4f}."
        if fine_tuning_required
        else f"Fine-tuning was not required because frozen-backbone feature extraction reached {validation_before:.4f}, at or above 0.8000."
    )
    report_path = OUTPUT_DIR / "part2_tasks_3_6_training_summary.md"
    report_path.write_text(
        f"""# Part 2 — Tasks 3–6 Training Summary

## Model and feature extraction

The model uses an ImageNet-pretrained ResNet-18 backbone with its original fully connected layer replaced by a new 10-output linear classifier for Fashion-MNIST. All backbone parameters were frozen during feature extraction. Train and validation feature vectors were cached in memory from one backbone pass each; the classifier head was then trained from those cached vectors, with no frozen-backbone recomputation during the {FEATURE_EXTRACTION_EPOCHS} head-only epochs.

- Device: `{device}`
- Input size: {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}
- Batch size: {BATCH_SIZE}
- Optimizer: Adam
- Feature-extraction learning rate: {FEATURE_EXTRACTION_LEARNING_RATE}
- Feature-extraction epochs: {FEATURE_EXTRACTION_EPOCHS}
- Validation accuracy before fine-tuning: {validation_before:.4f}

{fine_tuning_statement}

## Final held-out evaluation

Test accuracy: **{test_accuracy:.4f}**. The test split was evaluated only after the validation-based decision about fine-tuning.

The full confusion matrix is saved in `part2_tasks_3_6_confusion_matrix.csv`, and per-class precision and recall are saved in `part2_tasks_3_6_classification_report.csv`.

## Most frequent observed confusions

""" + "\n\n".join(
            f"### {row.actual_class} predicted as {row.predicted_class} ({row.misclassified_images} images)\n\n{row.explanation}"
            for row in confusion_pairs.itertuples()
        ) + "\n\nNo model weights or sample images were saved or exported in this Tasks 3–6 script.\n",
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Validation accuracy before fine-tuning: {validation_before:.4f}")
    print(f"Fine-tuning required: {fine_tuning_required}")
    print(f"Validation accuracy after fine-tuning: {validation_after:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(confusion_pairs[["actual_class", "predicted_class", "misclassified_images"]].to_string(index=False))
    print(f"Training summary: {report_path}")


if __name__ == "__main__":
    main()
