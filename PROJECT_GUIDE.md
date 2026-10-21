# Project guide

## The question

This project looks at a straightforward question: **how much information does median income give us about the median house value in a California census block group?**

It is an exercise in regression, not a production house-pricing service. The data describes areas rather than individual homes, so the results should not be interpreted as an estimate for a specific property.

## What the data contains

The dataset has eight input features:

- `MedInc`: median income.
- `HouseAge`: median house age.
- `AveRooms`: average rooms per household.
- `AveBedrms`: average bedrooms per household.
- `Population`: block-group population.
- `AveOccup`: average household occupancy.
- `Latitude` and `Longitude`: geographic location.

The target is `MedHouseVal`, measured in units of $100,000. The analysis uses all features to inspect the data and its correlations, but the two model experiments focus on `MedInc` so the relationship is easier to visualize.

## How the analysis works

### 1. Inspect the dataset

The script prints the dataset size, basic summary statistics, and a correlation matrix. The heatmap is a quick way to see which features move together. Correlation by itself does not establish cause and effect.

### 2. Build linear regression with NumPy

The first model is deliberately implemented without a scikit-learn estimator. It standardizes the income feature using the training-set mean and standard deviation, then repeatedly updates a weight and bias to reduce mean squared error.

For a single feature, the prediction has the form:

`prediction = weight × standardized_income + bias`

The training-cost curve helps show whether the optimization is reducing the error over time.

### 3. Compare polynomial models

A straight line may miss a curved relationship. Polynomial features let the model fit a curve while keeping the estimator linear in its learned coefficients. The script compares degrees 2, 3, and 5 against the linear baseline.

### 4. Evaluate fairly

The data is split into training and test sets before fitting. The test set is kept separate from training and is used to calculate:

- **MSE:** average squared prediction error; lower is better.
- **R²:** how much target variation the model explains relative to a simple mean baseline; higher is generally better.

The script writes these metrics to `outputs/model_metrics.csv`. The measured scores can vary if the split or random seed changes, so use the output from a run rather than quoting an unverified score.

## Important limitations

- Only median income is used as the input to the fitted models. Other potentially useful features are not included in the prediction experiment.
- The dataset describes census block groups, not individual homes.
- Some target values are capped near $500,000, which limits what the models can learn at the high end.
- A random train/test split is a useful baseline, but it does not test how well the model generalizes to a different time period or location.
- Better scores do not automatically make a model suitable for real estate decisions.

## Suggested improvements

1. Add a multivariable regression model using all eight features.
2. Compare against a simple mean-prediction baseline.
3. Use cross-validation to check how sensitive the results are to the split.
4. Inspect residuals and errors across different price ranges.
5. Try regularization or a tree-based model and compare results on the same test split.

## Files

- `src/regression_project.py` — runnable analysis and plotting code.
- `README.md` — setup, execution, and project overview.
- `requirements.txt` — Python dependencies.
