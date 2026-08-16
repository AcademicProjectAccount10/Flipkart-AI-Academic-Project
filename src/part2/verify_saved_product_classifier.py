"""Fresh-process verification for the saved Part 2 classifier artifact."""

import json
from pathlib import Path

from product_classifier import MODEL_PATH, load_product_classifier, predict_single_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "sample_images" / "0019_t-shirt_top.png"


def main() -> None:
    model = load_product_classifier(MODEL_PATH)
    result = predict_single_image(SAMPLE_IMAGE, MODEL_PATH)
    verification = {
        "model_path": str(MODEL_PATH),
        "model_loaded_in_fresh_process": True,
        "model_in_evaluation_mode": not model.training,
        "inference_image": str(SAMPLE_IMAGE),
        "inference_result": result,
        "valid_predicted_class": 0 <= result["predicted_index"] < 10,
        "valid_confidence": 0.0 <= result["confidence"] <= 1.0,
    }
    output_path = OUTPUT_DIR / "part2_tasks_7_8_fresh_process_verification.json"
    output_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
