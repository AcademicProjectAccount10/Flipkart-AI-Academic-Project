# Part 2 — Tasks 3–6 Training Summary

## Model and feature extraction

The model uses an ImageNet-pretrained ResNet-18 backbone with its original fully connected layer replaced by a new 10-output linear classifier for Fashion-MNIST. All backbone parameters were frozen during feature extraction. Train and validation feature vectors were cached in memory from one backbone pass each; the classifier head was then trained from those cached vectors, with no frozen-backbone recomputation during the 12 head-only epochs.

- Device: `cpu`
- Input size: 224 × 224
- Batch size: 128
- Optimizer: Adam
- Feature-extraction learning rate: 0.001
- Feature-extraction epochs: 12
- Validation accuracy before fine-tuning: 0.8962

Fine-tuning was not required because frozen-backbone feature extraction reached 0.8962, at or above 0.8000.

## Final held-out evaluation

Test accuracy: **0.8838**. The test split was evaluated only after the validation-based decision about fine-tuning.

The full confusion matrix is saved in `part2_tasks_3_6_confusion_matrix.csv`, and per-class precision and recall are saved in `part2_tasks_3_6_classification_report.csv`.

## Most frequent observed confusions

### T-shirt/top predicted as Shirt (156 images)

T-shirt/top and Shirt are both grayscale upper-body garment silhouettes. At 28×28 resolution, sleeves, collars, and fabric texture can be too faint to separate them reliably.

### Coat predicted as Shirt (112 images)

Coat and Shirt are both grayscale upper-body garment silhouettes. At 28×28 resolution, sleeves, collars, and fabric texture can be too faint to separate them reliably.

No model weights or sample images were saved or exported in this Tasks 3–6 script.
