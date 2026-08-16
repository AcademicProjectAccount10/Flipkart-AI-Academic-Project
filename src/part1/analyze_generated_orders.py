"""Verification and descriptive analysis for Part 1, Tasks 1 and 2."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "orders_dataset.csv"


def percentage(series: pd.Series) -> float:
    """Return a proportion as a percentage rounded to two decimals."""
    return round(series.mean() * 100, 2)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    expected_shape = (6000, 13)
    if df.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape}, found {df.shape}.")

    category_rates = (
        df.groupby("product_category", sort=True)["returned"]
        .mean()
        .mul(100)
        .round(2)
        .rename("return_rate_pct")
    )
    payment_rates = (
        df.groupby("payment_method", sort=True)["returned"]
        .mean()
        .mul(100)
        .round(2)
        .rename("return_rate_pct")
    )
    missing_rates_by_payment = (
        df.assign(rating_missing=df["rating_given"].isna())
        .groupby("payment_method", sort=True)["rating_missing"]
        .mean()
        .mul(100)
        .round(2)
    )
    cod_missing = percentage(
        df.loc[df["payment_method"].eq("COD"), "rating_given"].isna()
    )
    non_cod_missing = percentage(
        df.loc[~df["payment_method"].eq("COD"), "rating_given"].isna()
    )

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Overall return rate: {percentage(df['returned']):.2f}%")
    print(f"Missing rating_given: {percentage(df['rating_given'].isna()):.2f}%")
    print("\nReturn rate by product_category (%):")
    print(category_rates.to_string())
    print("\nReturn rate by payment_method (%):")
    print(payment_rates.to_string())
    print("\nMissing rating_given by payment_method (%):")
    print(missing_rates_by_payment.to_string())
    print(
        "\nMissingness mechanism: MAR. Missingness is conditioned on the observed "
        "payment_method variable in the generator, not on the unobserved rating itself. "
        f"The measured COD versus non-COD missing-rate gap is "
        f"{cod_missing - non_cod_missing:.2f} percentage points "
        f"({cod_missing:.2f}% vs {non_cod_missing:.2f}%)."
    )


if __name__ == "__main__":
    main()
