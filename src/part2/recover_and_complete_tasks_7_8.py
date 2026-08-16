"""One-time recovery for Part 2 Tasks 7--8 when prior in-memory weights were lost."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.optim import Adam

from product_classifier import CLASS_NAMES, MODEL_PATH, FashionResNet18, checkpoint_metadata
from setup_fashion_mnist import DATA_ROOT, IMAGE_SIZE, create_datasets_and_loaders
from train_tasks_3_to_6 import (
    BATCH_SIZE,
    FEATURE_EXTRACTION_EPOCHS,
    FEATURE_EXTRACTION_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    cache_frozen_features,
    cached_accuracy,
    evaluate_model,
    fine_tune_late_layer,
    train_classifier_head,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_images"
RANDOM_STATE = 42


def set_seed() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def export_verified_test_samples() -> list[dict[str, object]]:
    """Export one real official-test image for each of five different classes."""
    from torchvision.datasets import FashionMNIST

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    test_source = FashionMNIST(root=DATA_ROOT, train=False, download=True)
    targets = np.asarray(test_source.targets)
    chosen_labels = [0, 1, 2, 5, 7]
    records: list[dict[str, object]] = []
    for label in chosen_labels:
        index = int(np.flatnonzero(targets == label)[0])
        filename_label = CLASS_NAMES[label].lower().replace("/", "_").replace(" ", "_")
        output_path = SAMPLE_DIR / f"{index:04d}_{filename_label}.png"
        source_array = test_source.data[index].numpy()
        Image.fromarray(source_array, mode="L").save(output_path)
        with Image.open(output_path) as exported:
            exported_array = np.asarray(exported.convert("L"))
        verified = bool(np.array_equal(source_array, exported_array))
        if not verified:
            raise ValueError(f"PNG verification failed for test index {index}.")
        records.append(
            {
                "filename": output_path.name,
                "test_split_index": index,
                "true_label_index": int(targets[index]),
                "true_label": CLASS_NAMES[label],
                "verified_pixel_equal_to_official_test_sample": verified,
            }
        )
    return records


def main() -> None:
    set_seed()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = create_datasets_and_loaders(batch_size=BATCH_SIZE)

    # Minimum recovery: one frozen-backbone pass for train/validation, then head-only training.
    model = FashionResNet18(pretrained=True).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    train_cache = cache_frozen_features(model.backbone, data.train_loader, device)
    validation_cache = cache_frozen_features(model.backbone, data.validation_loader, device)
    _, trained_head = train_classifier_head(model.classifier, train_cache, device)
    model.classifier = trained_head
    validation_before = cached_accuracy(model.classifier, validation_cache, device)

    fine_tuning_required = validation_before < 0.80
    validation_after = validation_before
    if fine_tuning_required:
        _, validation_after = fine_tune_late_layer(model, data.train_loader, data.validation_loader, device)

    # The official test split is evaluated only after the validation-based choice above.
    test_accuracy, _, _ = evaluate_model(model, data.test_loader, device)
    model.eval()
    checkpoint = {
        **checkpoint_metadata(),
        "state_dict": model.cpu().state_dict(),
        "feature_extraction_learning_rate": FEATURE_EXTRACTION_LEARNING_RATE,
        "feature_extraction_epochs": FEATURE_EXTRACTION_EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_accuracy": validation_after,
        "test_accuracy": test_accuracy,
        "fine_tuning_required": fine_tuning_required,
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE if fine_tuning_required else None,
    }
    torch.save(checkpoint, MODEL_PATH)

    sample_records = export_verified_test_samples()
    pd.DataFrame(sample_records).to_csv(
        OUTPUT_DIR / "part2_tasks_7_8_exported_test_samples.csv", index=False
    )
    recovery_report = {
        "recovery_reason": "No trained checkpoint or cached feature vectors were persisted after Tasks 3--6.",
        "repeated_computation": "One frozen ResNet-18 pass for train/validation plus 12 head-only Adam epochs.",
        "fine_tuning_required": fine_tuning_required,
        "validation_accuracy": validation_after,
        "test_accuracy": test_accuracy,
        "model_path": str(MODEL_PATH),
        "sample_images": sample_records,
    }
    (OUTPUT_DIR / "part2_tasks_7_8_recovery_report.json").write_text(
        json.dumps(recovery_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(recovery_report, indent=2))


if __name__ == "__main__":
    main()
