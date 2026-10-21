"""Explore California housing values with linear and polynomial regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset from scikit-learn."""
    dataset = fetch_california_housing(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename("MedHouseVal")
    return features, target


def print_summary(features: pd.DataFrame, target: pd.Series) -> None:
    """Print a compact overview of the features and target."""
    print("\nDataset overview")
    print("----------------")
    print(f"Rows: {len(features):,}")
    print(f"Features: {features.shape[1]}")
    print(f"Missing feature values: {int(features.isna().sum().sum())}")
    print("\nFeature summary:")
    print(features.describe().round(2).T.to_string())
    print("\nTarget summary (units of $100,000):")
    print(target.describe().round(3).to_string())


def train_gradient_descent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.05,
    iterations: int = 3_000,
) -> tuple[float, float, list[float], float, float]:
    """Fit one-feature linear regression using batch gradient descent.

    The input is standardized from the training data only. The returned mean
    and standard deviation are needed to transform test and plotting inputs.
    """
    mean = float(np.mean(x_train))
    scale = float(np.std(x_train))
    if scale == 0:
        raise ValueError("The training feature has zero variance.")

    x_scaled = (x_train - mean) / scale
    weight = 0.0
    bias = float(np.mean(y_train))
    costs: list[float] = []

    for _ in range(iterations):
        predictions = weight * x_scaled + bias
        errors = predictions - y_train
        costs.append(float(np.mean(errors**2)))

        grad_weight = float(np.mean(errors * x_scaled))
        grad_bias = float(np.mean(errors))
        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

    return weight, bias, costs, mean, scale


def predict_gradient_descent(
    x: np.ndarray, weight: float, bias: float, mean: float, scale: float
) -> np.ndarray:
    """Predict values with the trained NumPy model."""
    return weight * ((x - mean) / scale) + bias


def fit_polynomial_models(
    x_train: np.ndarray, y_train: np.ndarray, degrees: tuple[int, ...] = (1, 2, 3, 5)
) -> dict[int, tuple[PolynomialFeatures, LinearRegression]]:
    """Fit polynomial regression models using the training split."""
    fitted: dict[int, tuple[PolynomialFeatures, LinearRegression]] = {}
    for degree in degrees:
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        train_features = transformer.fit_transform(x_train.reshape(-1, 1))
        model = LinearRegression()
        model.fit(train_features, y_train)
        fitted[degree] = (transformer, model)
    return fitted


def evaluate_models(
    y_test: np.ndarray,
    baseline_predictions: np.ndarray,
    polynomial_models: dict[int, tuple[PolynomialFeatures, LinearRegression]],
    x_test: np.ndarray,
) -> pd.DataFrame:
    """Return test-set metrics for the NumPy baseline and polynomial models."""
    rows = [
        {
            "model": "NumPy linear regression",
            "degree": 1,
            "mse": mean_squared_error(y_test, baseline_predictions),
            "r2": r2_score(y_test, baseline_predictions),
        }
    ]

    for degree, (transformer, model) in polynomial_models.items():
        predictions = model.predict(transformer.transform(x_test.reshape(-1, 1)))
        rows.append(
            {
                "model": f"Polynomial regression (degree {degree})",
                "degree": degree,
                "mse": mean_squared_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }
        )

    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def save_plots(
    features: pd.DataFrame,
    target: pd.Series,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: float,
    bias: float,
    costs: list[float],
    mean: float,
    scale: float,
    polynomial_models: dict[int, tuple[PolynomialFeatures, LinearRegression]],
) -> None:
    """Save the project's plots to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    correlation = features.assign(MedHouseVal=target).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("California Housing: Feature Correlations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(2_000, len(x_train))
    sample_indices = rng.choice(len(x_train), size=sample_size, replace=False)
    x_line = np.linspace(float(np.min(x_train)), float(np.max(x_train)), 250)
    y_line = predict_gradient_descent(x_line, weight, bias, mean, scale)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        x_train[sample_indices], y_train[sample_indices], s=10, alpha=0.25, label="Training data"
    )
    axes[0].plot(x_line, y_line, linewidth=2.5, label="NumPy linear model")
    axes[0].set_xlabel("Median income (MedInc)")
    axes[0].set_ylabel("Median house value ($100,000 units)")
    axes[0].set_title("Linear Regression with Gradient Descent")
    axes[0].legend()

    axes[1].plot(costs, linewidth=1.8)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Training MSE")
    axes[1].set_title("Gradient Descent Cost")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "linear_regression_gradient_descent.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        x_train[sample_indices], y_train[sample_indices], s=10, alpha=0.2, label="Training data"
    )
    for degree, (transformer, model) in polynomial_models.items():
        if degree == 1:
            continue
        predictions = model.predict(transformer.transform(x_line.reshape(-1, 1)))
        ax.plot(x_line, predictions, linewidth=2, label=f"Polynomial degree {degree}")
    ax.set_xlabel("Median income (MedInc)")
    ax.set_ylabel("Median house value ($100,000 units)")
    ax.set_title("Polynomial Regression Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "polynomial_regression_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Print metrics and save the CSV without generating figures.",
    )
    args = parser.parse_args()

    features, target = load_data()
    print_summary(features, target)

    x = features["MedInc"].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    weight, bias, costs, mean, scale = train_gradient_descent(x_train, y_train)
    numpy_predictions = predict_gradient_descent(x_test, weight, bias, mean, scale)

    polynomial_models = fit_polynomial_models(x_train, y_train)
    metrics = evaluate_models(y_test, numpy_predictions, polynomial_models, x_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\nTest-set model comparison")
    print("-------------------------")
    print(metrics.to_string(index=False, formatters={"mse": "{:.4f}".format, "r2": "{:.4f}".format}))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

    if not args.skip_plots:
        save_plots(
            features,
            target,
            x_train,
            y_train,
            weight,
            bias,
            costs,
            mean,
            scale,
            polynomial_models,
        )
        print("Plots saved in the outputs/ directory.")


if __name__ == "__main__":
    main()
