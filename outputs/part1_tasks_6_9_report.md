# Part 1 — Tasks 6–9 Results

## Random Forest tuning

Best parameters: `{'classifier__max_depth': 6, 'classifier__n_estimators': 200}`. Best 5-fold cross-validated ROC-AUC: 0.6192. Held-out test ROC-AUC: 0.6203.

## Top five impurity-based features

```
            transformed_feature       source_feature  impurity_importance  impurity_rank
categorical__payment_method_COD       payment_method               0.1788              1
             numeric__price_inr            price_inr               0.1323              2
  numeric__delivery_distance_km delivery_distance_km               0.0957              3
  numeric__customer_tenure_days customer_tenure_days               0.0900              4
         numeric__delivery_days        delivery_days               0.0884              5
```

Interpretation:
- `categorical__payment_method_COD`: Payment method is directly informative here because COD orders were assigned higher simulated return risk.
- `numeric__price_inr`: Price can affect the perceived cost of a mistaken purchase and differs materially across product types.
- `numeric__delivery_distance_km`: Distance can appear important to tree splits even when it has no causal role, so its held-out permutation result is essential.
- `numeric__customer_tenure_days`: Tenure proxies for customer familiarity and purchasing history, which can be associated with return behaviour.
- `numeric__delivery_days`: Longer delivery times may raise the chance that a customer no longer wants an order by delivery.

## Permutation comparison

The table below compares the top five impurity-ranked encoded features against permutation importance measured as the held-out ROC-AUC decrease after shuffling that same encoded column.

```
            transformed_feature  impurity_importance  impurity_rank  permutation_importance_mean  permutation_rank  rank_change
categorical__payment_method_COD               0.1788              1                       0.0651                 1            0
             numeric__price_inr               0.1323              2                       0.0124                 2            0
  numeric__delivery_distance_km               0.0957              3                       0.0006                10            7
  numeric__customer_tenure_days               0.0900              4                      -0.0051                18           14
         numeric__delivery_days               0.0884              5                       0.0030                 5            0
```

`numeric__customer_tenure_days` drops the most from its impurity rank (4) to its permutation rank (18). The original top-five features with substantial drops are: `numeric__delivery_distance_km`, `numeric__customer_tenure_days`. In particular, the continuous `numeric__delivery_distance_km` is not part of the return-generating process but receives considerable impurity importance. Impurity-based importance can overrate a noisy continuous feature because trees have many possible split points for it, creating impurity reductions by chance that do not improve held-out performance.

## Subgroup performance at the default 0.50 threshold

Overall precision: 0.3240; overall recall: 0.5495.

### By product category

```
product_category  test_rows  precision_class_1  recall_class_1
         Apparel   385.0000             0.3171          0.5200
          Beauty   116.0000             0.4750          0.6129
     Electronics   261.0000             0.3286          0.4423
        Footwear   217.0000             0.3626          0.5893
            Home   221.0000             0.2347          0.6765
```

### By payment method

```
payment_method  test_rows  precision_class_1  recall_class_1
           COD   503.0000             0.3273          0.9355
  Prepaid_Card   283.0000             0.2000          0.0204
   Prepaid_UPI   294.0000             0.3333          0.0417
        Wallet   120.0000             0.2222          0.0952
```

`Prepaid_Card` is the weakest payment subgroup, with recall of 0.0204 versus the overall 0.5495. A concrete next step is to select a lower, Prepaid_Card-specific decision threshold on a validation set, increasing recovery of likely returns for that payment method while measuring its added false-positive workload separately.

## Random Forest threshold sweep and artifact

The Random Forest's F1-maximizing threshold is **t*_rf = 0.50**, using this saved model's own held-out `predict_proba` output. At t*_rf, precision is 0.3240, recall is 0.5495, and F1 is 0.4076.

The fitted preprocessing-plus-Random-Forest pipeline was saved to `models/return_risk_model.pkl` and reloaded with `joblib.load()` successfully; the loaded model's first five probabilities matched the in-memory pipeline exactly.
