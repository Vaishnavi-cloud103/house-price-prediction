# House Price Prediction 🏠

Predicting house sale prices using the Ames Housing dataset (Kaggle: *House Prices - Advanced Regression Techniques*). Compared three regression models and improved prediction accuracy from **64% to 91% R²** through better modeling choices.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | $21,961 | $52,272 | 0.644 |
| Random Forest (300 trees) | $17,228 | $28,505 | 0.894 |
| **Gradient Boosting (final model)** | **$16,097** | **$26,038** | **0.912** |

**Final model:** Gradient Boosting Regressor — explains ~91% of the variance in house prices, with predictions typically within ~$16,000 of the actual sale price.

## Dataset

- 1,460 labeled training houses + 1,459 unlabeled test houses
- 80 features per house: size, quality ratings, neighborhood, year built, garage, basement, etc.
- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Approach

**1. Data cleaning**
Most "missing" values weren't errors — they meant a feature simply didn't apply (e.g. blank `PoolQC` = no pool). Filled accordingly with `"None"`/`0` rather than guessing values. `LotFrontage` was imputed using the median for each neighborhood rather than a global average.

**2. Feature encoding**
- **Ordinal encoding** for ordered categories (e.g. quality ratings Poor→Excellent → 0–5)
- **One-hot encoding** for unordered categories (e.g. `Neighborhood`)
- Train and test sets were combined *before* encoding to guarantee identical columns in both, avoiding train/test feature mismatches

**3. Modeling**
Compared three regressors, starting simple and increasing in complexity:
- **Linear Regression** — baseline
- **Random Forest** — ensemble of independently trained trees, averaged
- **Gradient Boosting** — trees trained sequentially, each correcting the previous trees' errors (best performer)

**4. Feature importance**
The strongest predictors of price were `OverallQual` (overall material/finish quality) and `GrLivArea` (living area square footage) — consistent with real-world intuition, which helps validate that the model learned genuine patterns rather than noise.

## Tech stack

- Python
- pandas, numpy
- scikit-learn (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`)

## Repository contents

| File | Description |
|---|---|
| `house_price_prediction.py` | Full end-to-end pipeline: clean → encode → train → evaluate → predict |
| `submission.csv` | Final predictions for the test set |

## How to run

```bash
pip install pandas numpy scikit-learn
python house_price_prediction.py
```

Requires `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) in the same directory.

## Key lessons learned

- "Missing" data often isn't broken data — understanding *why* it's missing matters before deciding how to handle it
- CSV round-trips can silently reintroduce missing values (pandas treats the literal text `"None"` as `NaN` by default when reading a file)
- Combining train and test data before one-hot encoding keeps feature columns consistent
- Always validate on unseen data — never judge a model using its own training data
- Tree-based ensemble methods handle mixed numeric/categorical data and skewed targets better than plain linear models

## Possible next steps

- Cross-validation instead of a single train/test split
- Log-transform `SalePrice` to reduce the effect of its right skew
- Hyperparameter tuning (`GridSearchCV`)
- Feature engineering (e.g. `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`)
