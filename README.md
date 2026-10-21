# California Housing: Regression From Scratch

A small machine-learning project exploring how median income relates to California house values. It starts with a linear regression model implemented with NumPy, then compares it with polynomial regression models built with scikit-learn.

The project is written as a regular Python script so it can be run from a terminal instead of requiring a Jupyter notebook.

## What it covers

- Load the California Housing dataset from scikit-learn.
- Inspect the features and summarize the data.
- Look at correlations between the available features.
- Implement gradient descent for a one-feature linear regression model using NumPy.
- Fit polynomial regression models of degrees 2, 3, and 5.
- Compare models on a held-out test set using mean squared error (MSE) and R².
- Save plots and a model comparison table locally.

## Dataset

The project uses the California Housing dataset provided by scikit-learn. It contains census-block-group information, not a record for every individual house. The target, `MedHouseVal`, is expressed in units of $100,000. For example, a target value of 2.5 corresponds to $250,000.

The dataset is downloaded by scikit-learn the first time it is needed, so the first run requires an internet connection. The dataset has known limitations, including a target ceiling around $500,000; predictions should be treated as an educational exercise, not as professional property valuations.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```powershell
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

From the repository root:

```bash
python src/regression_project.py
```

The script prints dataset summaries and evaluation metrics. It also creates an `outputs/` directory containing:

- `model_metrics.csv` — test-set results for the models.
- `correlation_heatmap.png` — feature correlation plot.
- `linear_regression_gradient_descent.png` — the NumPy model and its training-cost curve.
- `polynomial_regression_comparison.png` — fitted polynomial curves.

To run the analysis without creating plots:

```bash
python src/regression_project.py --skip-plots
```

## Models

1. **Linear regression from scratch** — uses median income as the single input feature and learns the weight and bias with gradient descent.
2. **Polynomial regression** — expands median income into polynomial features of degrees 2, 3, and 5, then fits a linear regression estimator.

All models are evaluated on the same held-out test split. The script reports the results rather than assuming a more complex model is automatically better.

## Project structure

```text
Project-in-Regression/
├── README.md
├── PROJECT_GUIDE.md
├── requirements.txt
├── .gitignore
└── src/
    └── regression_project.py
```

## Scope and next steps

This is a learning project focused on the relationship between one feature and the target. It does not yet use all eight features in a predictive model, tune hyperparameters, or validate performance across multiple splits. Useful next steps would be to add a multivariable baseline, cross-validation, and residual analysis.

## References

- [California Housing dataset in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Linear regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Polynomial features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
