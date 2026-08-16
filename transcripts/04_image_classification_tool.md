# Image-classification tool conversation

Mode: `MOCK_LLM_MODE=True` (local deterministic mode; no API key or live LLM).

## Turn 1
**User:** What product category is this image?

**Intent route:** `image_classification`

**Routing evidence:** Applied few-shot routing example: User: "What product category is this image?" with an image path supplied → Intent: image_classification

**Real tool output:**
```json
{
  "image_path": "data\\sample_images\\0019_t-shirt_top.png",
  "predicted_index": 0,
  "predicted_label": "T-shirt/top",
  "confidence": 0.9875097274780273
}
```

**Assistant JSON response:**
```json
{
  "answer": "The image is predicted as T-shirt/top with 98.75% confidence.",
  "source": "image_classifier_tool",
  "confidence": 0.9875
}
```
