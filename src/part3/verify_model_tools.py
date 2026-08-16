"""Direct parity checks for Part 3 Tasks 3--4 model tools; no agent is built here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from model_tools import RETURN_RISK_FEATURES, check_return_risk, classify_product_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "part3_tasks_3_4_tool_verification.json"
PART2_SOURCE_DIR = PROJECT_ROOT / "src" / "part2"


def main() -> None:
    order = pd.read_csv(PROJECT_ROOT / "orders_dataset.csv").iloc[0]
    order_features = {feature: order[feature].item() if hasattr(order[feature], "item") else order[feature] for feature in RETURN_RISK_FEATURES}
    risk_result = check_return_risk(order_features)
    direct_probability = float(
        joblib.load(PROJECT_ROOT / "models" / "return_risk_model.pkl")
        .predict_proba(pd.DataFrame([order_features]))[0, 1]
    )

    image_path = PROJECT_ROOT / "data" / "sample_images" / "0019_t-shirt_top.png"
    image_result = classify_product_image(str(image_path))
    if str(PART2_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(PART2_SOURCE_DIR))
    from product_classifier import predict_single_image

    direct_image_result = predict_single_image(image_path)
    verification = {
        "return_risk_tool": risk_result,
        "return_risk_direct_probability": direct_probability,
        "return_risk_matches_saved_model": risk_result["return_probability"] == direct_probability,
        "image_tool": image_result,
        "image_direct_model_result": direct_image_result,
        "image_tool_matches_saved_model": {
            key: image_result[key] == direct_image_result[key]
            for key in ("predicted_index", "predicted_label", "confidence")
        },
    }
    OUTPUT_PATH.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
