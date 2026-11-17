# Week 04: Compare Many Models
 
**Project:** Predicting House Prices
**Notebook:** `notebooks/04_compare_models.ipynb`
 
This week you try about ten models and compare them fairly using cross-validation.
 
**By the end of this week you will be able to:**
 
- Explain cross-validation
- Use a dictionary and a loop to train many models
- Explain why some models need feature scaling
- 
Plan for about ~ 1.5 hours.
 
---
 
## New Idea: Cross-Validation
 
One train/validation split can be lucky or unlucky. With only 58 validation houses, a few strange houses can change the score a lot.
 
**Cross-validation (CV)** cuts the training data into 5 parts, trains on 4, scores on the 5th, and repeats so every part gets a turn. The average of the 5 scores is more reliable, and the std shows how much the score jumps around. It never touches the validation set or the test file.
 
## New Idea: Scaling
 
Some models compare features by size. An `age_sq` that goes above 400 would drown out a store count (0 to 10). A `StandardScaler` puts every feature on the same scale. Linear models, KNN and SVR need it; tree models do not.
 
---
 
## Step 1: Set Up
 
- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `04_compare_models`
Copy your setup cell from Week 03 (with the outlier removal) to the top, and add one `cv` that every model will share:
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
 
df = pd.read_csv("../data/processed/houses_features.csv")
df = df[df["price"] <= 103.6]   # your exact cutoff from Week 02
 
FEATURES = [c for c in df.columns if c not in ("price", "trans_date")]
X = df[FEATURES]
y = df["price"]
 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```
 
---
 
## Step 2: Build a Dictionary of Models

![dictionary](images/dictionary.jpeg)
 
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
 
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    "SVR": make_pipeline(StandardScaler(), SVR(C=50)),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=500, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.03, verbose=-1, random_state=42),
}
```
 
If you get `ModuleNotFoundError` for `xgboost` or `lightgbm`, activate `.venv` and run `pip install -r requirements.txt` again.
 
---
 
## Step 3: Your Turn - Write the Loop
 
For each model in the dictionary:
 
1. Run `cross_validate(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")`. Average the scores and flip the sign to get **CV RMSE**, and save the std too. scikit-learn reports errors as negative numbers so that "higher is better" always holds.
2. Fit the model on `X_train`, `y_train`, predict `X_val`, and compute **validation RMSE** and **validation R²**.
3. Save each model's results as a dictionary in a list, then turn the list into a table with `pd.DataFrame(...)`.
> **Hint**
> 1. Make an empty list before the loop like `rows = []`
> 2. Inside the loop, do steps 1 and 2, then `rows.append({"model": name, "CV RMSE": ..., "CV std": ..., "val RMSE": ..., "val R2": ...})`
> 3. `mean_squared_error` and `r2_score` come from `sklearn.metrics`, like in Week 03
> 4. After the loop: `results = pd.DataFrame(rows).sort_values("CV RMSE")`
 
Draw a horizontal bar chart of it:
 
```python
results.plot(x="model", y="CV RMSE", kind="barh", xerr="CV std", legend=False)
plt.xlabel("CV RMSE (lower is better)")
plt.tight_layout()
plt.savefig("../reports/figures/model_comparison.png")
plt.show()
```
 
**Questions to answer in writing:**
 
1. Which three models have the lowest CV RMSE?
2. Do tree-based models beat linear ones? Why might that be, given the curves you saw in Week 01?
3. Is the gap between the best model and linear regression bigger or smaller than the CV std? What does that mean?
4. Are the CV scores and validation scores similar? What would it mean if they were very different?
---
 
## Deliverable
 
`04_compare_models.ipynb`, containing:
 
- [ ] At least 8 models trained in a loop
- [ ] A results table with CV RMSE, CV std, validation RMSE and validation R²
- [ ] A bar chart comparing the models, saved to `reports/figures/`
- [ ] Written answers to the questions
---
 
## Common Mistakes
 
- **Choosing the winner by validation score.** Use CV RMSE; 58 houses is a small practice exam.
- **Ignoring the std.** A win smaller than the spread between folds may just be noise.
- **Scaling tree models.** It doesn't hurt, but it isn't needed. Know why.
- **Looking for test file.** Ignore it until Week 06.
---
 
## Answers
 
- CV RMSE (lowest first):
  - Extra Trees ≈ 6.70
  - Random Forest ≈ 6.86
  - XGBoost ≈ 6.99
  - LightGBM ≈ 7.06
  - Gradient Boosting ≈ 7.08
  - Lasso, Ridge, Linear Regression ≈ 7.3
  - SVR ≈ 7.45
  - KNN ≈ 7.61
- Validation: tree models RMSE ≈ 6.4-6.9, R² ≈ 0.78-0.81. Linear models RMSE ≈ 7.3, R² ≈ 0.75.
- Tree models win because they can follow curves and U-shapes without being told about them.
- But the best model beats linear regression by only about 0.6, and the CV std is about 1. With 231 training houses, the top models are close to a tie.
- On validation, Random Forest is best (≈ 6.38), not Extra Trees. That's the lucky-split problem, which is why we pick by CV.
---
 
## Next Week
 
In Week 05, you'll tune the best models and combine them.
 
