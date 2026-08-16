"""Reusable architecture, loading, and single-image inference for Part 2."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models

from setup_fashion_mnist import IMAGE_SIZE, build_preprocessing_transform


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "product_classifier.pt"


class FashionResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 with a Fashion-MNIST 10-class head."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, len(CLASS_NAMES))

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.backbone(images))


def load_product_classifier(
    model_path: Path | str = MODEL_PATH, device: torch.device | str = "cpu"
) -> FashionResNet18:
    """Load the saved Task 7 artifact and return it in evaluation mode."""
    resolved_device = torch.device(device)
    checkpoint: dict[str, Any] = torch.load(
        Path(model_path), map_location=resolved_device, weights_only=True
    )
    if checkpoint.get("architecture") != "FashionResNet18":
        raise ValueError("Unexpected product-classifier architecture metadata.")
    if checkpoint.get("num_classes") != len(CLASS_NAMES):
        raise ValueError("Unexpected product-classifier class count.")
    model = FashionResNet18(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()
    return model


def predict_single_image(
    image_path: Path | str,
    model_path: Path | str = MODEL_PATH,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Load the saved model and classify one real image file."""
    resolved_device = torch.device(device)
    model = load_product_classifier(model_path=model_path, device=resolved_device)
    with Image.open(image_path) as image:
        input_tensor = build_preprocessing_transform()(image.convert("L")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor.to(resolved_device)), dim=1)[0].cpu()
    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_index": predicted_index,
        "predicted_label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
    }


def checkpoint_metadata() -> dict[str, object]:
    """Return the fixed reconstruction metadata saved alongside the weights."""
    return {
        "architecture": "FashionResNet18",
        "backbone": "ImageNet-pretrained ResNet-18",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "input_image_size": list(IMAGE_SIZE),
    }
