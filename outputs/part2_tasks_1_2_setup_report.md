# Part 2 — Tasks 1–2 Setup

## Dataset source

The dataset is the official Fashion-MNIST benchmark from Zalando Research, loaded exclusively with `torchvision.datasets.FashionMNIST(root=data/raw, download=True)`. It contains the official 60,000-image training split and 10,000-image test split.

## Stratified split sizes

| Split | Images | Source |
|---|---:|---|
| Training | 55,000 | Official training split |
| Validation | 5,000 | Stratified split from official training data |
| Test | 10,000 | Official untouched test split |

The validation split uses `StratifiedShuffleSplit(test_size=5000, random_state=42)` over only the official training labels. The test set is created separately and is not used for training or validation.

## Preprocessing for a pretrained backbone

Chosen backbone: **ResNet-18**. Required input image size: **224 × 224** pixels.

The preprocessing pipeline is: replicate the original one-channel grayscale image to three channels with `Grayscale(num_output_channels=3)`, resize to 224 × 224, convert to a tensor, then normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. This script creates reusable PyTorch `Dataset` and `DataLoader` objects only; it does not construct, train, fine-tune, evaluate, save, or export a model.
