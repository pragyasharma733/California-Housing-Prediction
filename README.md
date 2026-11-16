# California Housing Price Prediction

An end-to-end regression workflow predicting median house values in California districts. It compares linear, tree-based and regularised polynomial models, with cross-validation and hyperparameter tuning.

## Question
How accurately can we predict a district's median house value from census data, and which model gives the best balance between underfitting and overfitting?

## Data
The **California Housing** dataset (1990 census; the version from *Hands-On Machine Learning*) has 20,640 districts. The notebook downloads it automatically. Features:
- longitude and latitude
- housing median age
- total rooms and bedrooms
- population and households
- median income
- ocean proximity (categorical)

The target is `median_house_value`.

## Approach
1. **Stratified 80/20 split** on income category, so both sets reflect the income distribution.
2. **Preprocessing:**
   - median imputation for the 158 rows missing `total_bedrooms`
   - one-hot encoding for `ocean_proximity`
   - standard scaling
   - all combined in a single `ColumnTransformer` pipeline
3. **Models compared (10-fold CV):**
   - linear regression
   - decision tree
   - random forest
4. **Tuning:** `GridSearchCV` and `RandomizedSearchCV` over the random forest's `bootstrap`, `n_estimators` and `max_depth`.
5. **Bias–variance check:** learning curves for linear, degree-3 polynomial and ridge regression.

## Results
| Model | RMSE |
|---|---|
| Linear regression (10-fold CV) | $69,200 |
| Decision tree (train) | $0, which is overfitting |
| Tuned random forest (4-fold CV) | $50,300 |
| **Tuned random forest (test set)** | **$47,100** |

- **Linear regression underfits:** training and validation errors are both high (~$69k).
- **Degree-3 polynomial regression** lowers training error, but the gap to validation error suggests overfitting.
- **Ridge regression** on polynomial features closes that gap (~$60k train vs ~$61k validation).

## How to run
```bash
pip install numpy pandas scikit-learn scipy matplotlib six
jupyter notebook "California Housing Prediction.ipynb"
```

## Limitations and next steps
- The final test-set evaluation cell is duplicated. Remove one copy.
- Add engineered features such as rooms per household, bedrooms per room and population per household. These are known to help noticeably on this dataset.
- Try gradient boosting (e.g. `HistGradientBoostingRegressor`) and report feature importances.
- Note that the target is capped at $500,001. Many districts sit at the cap, which limits accuracy at the top end.
- `from six.moves import urllib` and `get_feature_names` are deprecated. Use `urllib.request` and `get_feature_names_out`.
