# Part 1 — Tasks 3–5 Results

## Split and preprocessing

The data was split into 4800 training rows and 1200 test rows using a stratified 80/20 split with `random_state=42`. Numeric features were median-imputed and standard-scaled. Categorical features were mode-imputed and one-hot encoded. Each preprocessing transformer was fitted only through its model pipeline's training-set `.fit(X_train, y_train)` call; the test set was only transformed during prediction.

## DummyClassifier baseline

Accuracy: 0.7725; F1 for `returned=1`: 0.0000.

The baseline predicts every order as not returned, so its apparently high accuracy reflects the majority class rather than useful detection. This is the **high accuracy, zero recall** failure mode: it correctly labels many non-returns but identifies none of the actual returns, making it unsuitable for proactive return-risk support.

## Logistic Regression at threshold 0.50

Accuracy: 0.5917; Precision: 0.2964; Recall: 0.5788; F1: 0.3921; ROC-AUC: 0.6253.

## F1-maximizing threshold

Best threshold: 0.44; Precision: 0.2801; Recall: 0.7582; F1: 0.4091.

Changing from 0.50 to 0.44 raises recall by 17.95 percentage points and changes precision by -1.63 percentage points. This threshold change makes missed returns (false negatives) more expensive to avoid, while accepting more false-positive return-risk flags and the additional support effort they cause.
