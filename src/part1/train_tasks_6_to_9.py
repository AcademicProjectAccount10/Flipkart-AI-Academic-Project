"""Part 1, Tasks 6--9: tuned Random Forest evaluation and model persistence."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from train_tasks_3_to_5 import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"


FEATURE_EXPLANATIONS = {
    "price_inr": "Price can affect the perceived cost of a mistaken purchase and differs materially across product types.",
    "discount_pct": "Large discounts can encourage more speculative purchases, which may be returned if the product is unsuitable.",
    "customer_tenure_days": "Tenure proxies for customer familiarity and purchasing history, which can be associated with return behaviour.",
    "num_previous_orders": "Past order volume provides context for interpreting a shopper's historical return count.",
    "num_previous_returns": "A larger number of prior returns is direct evidence of a customer's previous return behaviour.",
    "delivery_distance_km": "Distance can appear important to tree splits even when it has no causal role, so its held-out permutation result is essential.",
    "delivery_days": "Longer delivery times may raise the chance that a customer no longer wants an order by delivery.",
    "is_weekend_order": "Order timing can capture small differences in shopping intent and fulfilment conditions.",
    "rating_given": "Ratings may reflect engagement, although missing values are imputed and the field was not used to generate returns.",
    "product_category": "Category captures product-fit and product-type differences; Apparel and Footwear were assigned higher simulated risk.",
    "payment_method": "Payment method is directly informative here because COD orders were assigned higher simulated return risk.",
}


def source_feature_name(transformed_name: str) -> str:
    """Map a transformed feature name back to its original raw feature."""
    name = transformed_name.split("__", maxsplit=1)[-1]
    for categorical in CATEGORICAL_FEATURES:
        if name.startswith(f"{categorical}_"):
            return categorical
    return name


def subgroup_metrics(
    X_test: pd.DataFrame, y_test: pd.Series, predictions: pd.Series, column: str
) -> pd.DataFrame:
    """Return held-out precision and recall for each value of a subgroup column."""
    evaluation = X_test[[column]].copy()
    evaluation["actual_returned"] = y_test.to_numpy()
    evaluation["predicted_returned"] = predictions
    return (
        evaluation.groupby(column, sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "test_rows": len(group),
                    "precision_class_1": precision_score(
                        group["actual_returned"], group["predicted_returned"], zero_division=0
                    ),
                    "recall_class_1": recall_score(
                        group["actual_returned"], group["predicted_returned"], zero_division=0
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid={
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [6, 10, None],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = best_pipeline.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, test_probabilities)

    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_classifier = best_pipeline.named_steps["classifier"]
    transformed_feature_names = fitted_preprocessor.get_feature_names_out()
    impurity_importances = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "source_feature": [source_feature_name(name) for name in transformed_feature_names],
            "impurity_importance": fitted_classifier.feature_importances_,
        }
    ).sort_values("impurity_importance", ascending=False, ignore_index=True)
    impurity_importances["impurity_rank"] = impurity_importances.index + 1

    # Permute the same transformed columns used by the fitted Random Forest.
    X_test_transformed = fitted_preprocessor.transform(X_test)
    permutation = permutation_importance(
        fitted_classifier,
        X_test_transformed,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    permutation_importances = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "source_feature": [source_feature_name(name) for name in transformed_feature_names],
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)
    permutation_importances["permutation_rank"] = permutation_importances.index + 1

    importance_comparison = impurity_importances.merge(
        permutation_importances[
            [
                "transformed_feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "permutation_rank",
            ]
        ],
        on="transformed_feature",
        how="left",
    )
    top_five_comparison = importance_comparison.head(5).copy()
    top_five_comparison["rank_change"] = (
        top_five_comparison["permutation_rank"] - top_five_comparison["impurity_rank"]
    )

    category_subgroups = subgroup_metrics(X_test, y_test, test_predictions, "product_category")
    payment_subgroups = subgroup_metrics(X_test, y_test, test_predictions, "payment_method")
    overall_precision = precision_score(y_test, test_predictions, zero_division=0)
    overall_recall = recall_score(y_test, test_predictions, zero_division=0)

    thresholds = [round(value / 100, 2) for value in range(10, 91, 2)]
    threshold_rows = []
    for threshold in thresholds:
        threshold_predictions = (test_probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision_class_1": precision_score(y_test, threshold_predictions, zero_division=0),
                "recall_class_1": recall_score(y_test, threshold_predictions, zero_division=0),
                "f1_class_1": f1_score(y_test, threshold_predictions, zero_division=0),
            }
        )
    threshold_results = pd.DataFrame(threshold_rows)
    best_threshold = threshold_results.loc[threshold_results["f1_class_1"].idxmax()]

    pd.DataFrame(
        [
            {
                "best_params": str(grid_search.best_params_),
                "best_cross_validated_roc_auc": grid_search.best_score_,
                "held_out_test_roc_auc": test_roc_auc,
                "overall_precision_class_1": overall_precision,
                "overall_recall_class_1": overall_recall,
                "t_star_rf": best_threshold["threshold"],
                "f1_at_t_star_rf": best_threshold["f1_class_1"],
                "precision_at_t_star_rf": best_threshold["precision_class_1"],
                "recall_at_t_star_rf": best_threshold["recall_class_1"],
            }
        ]
    ).to_csv(OUTPUT_DIR / "part1_tasks_6_9_model_metrics.csv", index=False)
    impurity_importances.to_csv(OUTPUT_DIR / "part1_rf_impurity_importances.csv", index=False)
    permutation_importances.to_csv(OUTPUT_DIR / "part1_rf_permutation_importances.csv", index=False)
    top_five_comparison.to_csv(OUTPUT_DIR / "part1_rf_importance_comparison_top5.csv", index=False)
    category_subgroups.to_csv(OUTPUT_DIR / "part1_rf_subgroups_product_category.csv", index=False)
    payment_subgroups.to_csv(OUTPUT_DIR / "part1_rf_subgroups_payment_method.csv", index=False)
    threshold_results.to_csv(OUTPUT_DIR / "part1_rf_threshold_sweep.csv", index=False)

    joblib.dump(best_pipeline, MODEL_PATH)
    loaded_pipeline = joblib.load(MODEL_PATH)
    if not isinstance(loaded_pipeline, Pipeline):
        raise TypeError("Loaded artifact is not a scikit-learn Pipeline.")
    loaded_probabilities = loaded_pipeline.predict_proba(X_test.iloc[:5])[:, 1]
    if not pd.Series(loaded_probabilities).equals(pd.Series(test_probabilities[:5])):
        raise ValueError("Loaded pipeline predictions do not match the saved pipeline.")

    dropped_features = top_five_comparison.sort_values("rank_change", ascending=False)
    most_dropped = dropped_features.iloc[0]
    substantially_dropped = top_five_comparison.loc[
        top_five_comparison["rank_change"] >= 5, "transformed_feature"
    ].tolist()
    low_recall_payment = payment_subgroups.loc[payment_subgroups["recall_class_1"].idxmin()]
    impurity_table_text = top_five_comparison[
        ["transformed_feature", "source_feature", "impurity_importance", "impurity_rank"]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    comparison_table_text = top_five_comparison[
        [
            "transformed_feature",
            "impurity_importance",
            "impurity_rank",
            "permutation_importance_mean",
            "permutation_rank",
            "rank_change",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    category_table_text = category_subgroups.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
    payment_table_text = payment_subgroups.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
    feature_interpretations = "\n".join(
        f"- `{row.transformed_feature}`: {FEATURE_EXPLANATIONS.get(row.source_feature, 'This feature may capture a pattern associated with return risk.')}"
        for row in top_five_comparison.itertuples()
    )
    report = f"""# Part 1 — Tasks 6–9 Results

## Random Forest tuning

Best parameters: `{grid_search.best_params_}`. Best 5-fold cross-validated ROC-AUC: {grid_search.best_score_:.4f}. Held-out test ROC-AUC: {test_roc_auc:.4f}.

## Top five impurity-based features

```
{impurity_table_text}
```

Interpretation:
{feature_interpretations}

## Permutation comparison

The table below compares the top five impurity-ranked encoded features against permutation importance measured as the held-out ROC-AUC decrease after shuffling that same encoded column.

```
{comparison_table_text}
```

`{most_dropped.transformed_feature}` drops the most from its impurity rank ({most_dropped.impurity_rank}) to its permutation rank ({most_dropped.permutation_rank}). The original top-five features with substantial drops are: {", ".join(f'`{feature}`' for feature in substantially_dropped)}. In particular, the continuous `numeric__delivery_distance_km` is not part of the return-generating process but receives considerable impurity importance. Impurity-based importance can overrate a noisy continuous feature because trees have many possible split points for it, creating impurity reductions by chance that do not improve held-out performance.

## Subgroup performance at the default 0.50 threshold

Overall precision: {overall_precision:.4f}; overall recall: {overall_recall:.4f}.

### By product category

```
{category_table_text}
```

### By payment method

```
{payment_table_text}
```

`{low_recall_payment.payment_method}` is the weakest payment subgroup, with recall of {low_recall_payment.recall_class_1:.4f} versus the overall {overall_recall:.4f}. A concrete next step is to select a lower, Prepaid_Card-specific decision threshold on a validation set, increasing recovery of likely returns for that payment method while measuring its added false-positive workload separately.

## Random Forest threshold sweep and artifact

The Random Forest's F1-maximizing threshold is **t*_rf = {best_threshold['threshold']:.2f}**, using this saved model's own held-out `predict_proba` output. At t*_rf, precision is {best_threshold['precision_class_1']:.4f}, recall is {best_threshold['recall_class_1']:.4f}, and F1 is {best_threshold['f1_class_1']:.4f}.

The fitted preprocessing-plus-Random-Forest pipeline was saved to `models/return_risk_model.pkl` and reloaded with `joblib.load()` successfully; the loaded model's first five probabilities matched the in-memory pipeline exactly.
"""
    (OUTPUT_DIR / "part1_tasks_6_9_report.md").write_text(report, encoding="utf-8")

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Held-out test ROC-AUC: {test_roc_auc:.4f}")
    print("\nTop five impurity importances:")
    print(top_five_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTop five permutation importances:")
    print(permutation_importances.head(5).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nProduct-category subgroups:")
    print(category_subgroups.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPayment-method subgroups:")
    print(payment_subgroups.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nt*_rf: {best_threshold['threshold']:.2f}")
    print(f"Loaded artifact verified: {MODEL_PATH}")


if __name__ == "__main__":
    main()
