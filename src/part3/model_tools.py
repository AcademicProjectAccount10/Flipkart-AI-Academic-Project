"""Part 3 Tasks 3--4: real model-backed return-risk and image tools."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETURN_RISK_MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"
PART1_METRICS_PATH = PROJECT_ROOT / "outputs" / "part1_tasks_6_9_model_metrics.csv"
PART2_SOURCE_DIR = PROJECT_ROOT / "src" / "part2"
RETURN_RISK_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
    "product_category",
    "payment_method",
]


@lru_cache(maxsize=1)
def load_return_risk_model():
    """Load the tuned Part 1 Random Forest pipeline from its persisted artifact."""
    return joblib.load(RETURN_RISK_MODEL_PATH)


@lru_cache(maxsize=1)
def get_t_star_rf() -> float:
    """Read the F1-maximizing threshold from Part 1's Random Forest report."""
    metrics = pd.read_csv(PART1_METRICS_PATH)
    return float(metrics.loc[0, "t_star_rf"])


def check_return_risk(order_features: dict) -> dict[str, object]:
    """Return a real Random Forest return probability and t*_rf-anchored bucket."""
    missing_features = [feature for feature in RETURN_RISK_FEATURES if feature not in order_features]
    if missing_features:
        raise ValueError(f"Missing required order features: {missing_features}")
    feature_frame = pd.DataFrame([{feature: order_features[feature] for feature in RETURN_RISK_FEATURES}])
    probability = float(load_return_risk_model().predict_proba(feature_frame)[0, 1])
    t_star_rf = get_t_star_rf()
    high_cut_point = t_star_rf + 0.15
    if probability < t_star_rf:
        risk_bucket = "Low"
    elif probability < high_cut_point:
        risk_bucket = "Medium"
    else:
        risk_bucket = "High"
    return {
        "return_probability": probability,
        "risk_bucket": risk_bucket,
        "t_star_rf": t_star_rf,
        "low_cut_point": t_star_rf,
        "high_cut_point": high_cut_point,
    }


def classify_product_image(image_path: str) -> dict[str, object]:
    """Load and call the real saved Part 2 product classifier for an image file."""
    if str(PART2_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(PART2_SOURCE_DIR))
    from product_classifier import predict_single_image

    result = predict_single_image(Path(image_path))
    return {"image_path": str(Path(image_path)), **result}
