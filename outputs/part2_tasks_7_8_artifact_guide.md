# Part 2 — Tasks 7–8 Artifact Guide

## Saved model

The recovered Fashion-MNIST classifier is saved at `models/product_classifier.pt`. It is a self-describing PyTorch checkpoint containing the `FashionResNet18` state dictionary, ResNet-18 architecture metadata, class names, input size, and recovery training metadata. The architecture and reusable loading/inference functions are in `src/part2/product_classifier.py`.

## Load and predict one image

```python
from pathlib import Path
import sys

sys.path.insert(0, "src/part2")
from product_classifier import predict_single_image

result = predict_single_image(
    Path("data/sample_images/0019_t-shirt_top.png"),
    Path("models/product_classifier.pt"),
)
print(result)
# {'predicted_index': 0, 'predicted_label': 'T-shirt/top', 'confidence': ...}
```

`predict_single_image` reconstructs the saved model in evaluation mode, applies the ResNet-18 preprocessing used in Part 2 (three channels, 224×224 resize, ImageNet normalization), and returns the predicted label plus confidence.

## Exported test samples

Five actual Fashion-MNIST samples from the official untouched test split are saved in `data/sample_images/`. `part2_tasks_7_8_exported_test_samples.csv` records each source test index and verifies that its exported PNG pixels exactly equal the original official test image.
