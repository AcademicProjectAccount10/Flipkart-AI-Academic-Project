"""Part 1, Tasks 3--5: leakage-safe preprocessing and baseline models."""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "orders_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["product_category", "payment_method"]
NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]
TARGET = "returned"


def build_preprocessor() -> ColumnTransformer:
    """Create the required preprocessing transformer without fitting it."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def class_one_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate the requested held-out classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_class_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Each pipeline fits its preprocessor during .fit(X_train, y_train) only.
    dummy_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    dummy_pipeline.fit(X_train, y_train)
    dummy_predictions = dummy_pipeline.predict(X_test)
    dummy_metrics = class_one_metrics(y_test, dummy_predictions)

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(build_preprocessor())),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    # The fitted transformer is applied to test features only after training.
    logistic_predictions = logistic_pipeline.predict(X_test)
    logistic_probabilities = logistic_pipeline.predict_proba(X_test)[:, 1]
    logistic_metrics = class_one_metrics(y_test, logistic_predictions)
    logistic_metrics["roc_auc"] = roc_auc_score(y_test, logistic_probabilities)

    thresholds = [round(value, 2) for value in pd.Series(range(10, 91, 2)) / 100]
    threshold_rows = []
    for threshold in thresholds:
        threshold_predictions = (logistic_probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision_class_1": precision_score(
                    y_test, threshold_predictions, zero_division=0
                ),
                "recall_class_1": recall_score(y_test, threshold_predictions, zero_division=0),
                "f1_class_1": f1_score(y_test, threshold_predictions, zero_division=0),
            }
        )
    threshold_results = pd.DataFrame(threshold_rows)
    best_threshold_result = threshold_results.loc[
        threshold_results["f1_class_1"].idxmax()
    ]

    metrics_table = pd.DataFrame(
        [
            {"model": "DummyClassifier_most_frequent", **dummy_metrics},
            {"model": "LogisticRegression_default_0.50", **logistic_metrics},
            {
                "model": "LogisticRegression_best_f1_threshold",
                "accuracy": accuracy_score(
                    y_test, (logistic_probabilities >= best_threshold_result["threshold"]).astype(int)
                ),
                "precision_class_1": best_threshold_result["precision_class_1"],
                "recall_class_1": best_threshold_result["recall_class_1"],
                "f1_class_1": best_threshold_result["f1_class_1"],
                "roc_auc": logistic_metrics["roc_auc"],
                "threshold": best_threshold_result["threshold"],
            },
        ]
    )
    metrics_table.to_csv(OUTPUT_DIR / "part1_tasks_3_5_metrics.csv", index=False)
    threshold_results.to_csv(OUTPUT_DIR / "part1_logistic_threshold_sweep.csv", index=False)

    default_precision = logistic_metrics["precision_class_1"]
    best_precision = best_threshold_result["precision_class_1"]
    default_recall = logistic_metrics["recall_class_1"]
    best_recall = best_threshold_result["recall_class_1"]
    report = f"""# Part 1 — Tasks 3–5 Results

## Split and preprocessing

The data was split into {len(X_train)} training rows and {len(X_test)} test rows using a stratified 80/20 split with `random_state=42`. Numeric features were median-imputed and standard-scaled. Categorical features were mode-imputed and one-hot encoded. Each preprocessing transformer was fitted only through its model pipeline's training-set `.fit(X_train, y_train)` call; the test set was only transformed during prediction.

## DummyClassifier baseline

Accuracy: {dummy_metrics['accuracy']:.4f}; F1 for `returned=1`: {dummy_metrics['f1_class_1']:.4f}.

The baseline predicts every order as not returned, so its apparently high accuracy reflects the majority class rather than useful detection. This is the **high accuracy, zero recall** failure mode: it correctly labels many non-returns but identifies none of the actual returns, making it unsuitable for proactive return-risk support.

## Logistic Regression at threshold 0.50

Accuracy: {logistic_metrics['accuracy']:.4f}; Precision: {default_precision:.4f}; Recall: {default_recall:.4f}; F1: {logistic_metrics['f1_class_1']:.4f}; ROC-AUC: {logistic_metrics['roc_auc']:.4f}.

## F1-maximizing threshold

Best threshold: {best_threshold_result['threshold']:.2f}; Precision: {best_precision:.4f}; Recall: {best_recall:.4f}; F1: {best_threshold_result['f1_class_1']:.4f}.

Changing from 0.50 to {best_threshold_result['threshold']:.2f} raises recall by {(best_recall - default_recall) * 100:.2f} percentage points and changes precision by {(best_precision - default_precision) * 100:.2f} percentage points. This threshold change makes missed returns (false negatives) more expensive to avoid, while accepting more false-positive return-risk flags and the additional support effort they cause.
"""
    (OUTPUT_DIR / "part1_tasks_3_5_report.md").write_text(report, encoding="utf-8")

    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest threshold result:")
    print(best_threshold_result.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
