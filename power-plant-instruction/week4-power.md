# Week 04: Tune the Model and Try New Features

**Project:** Predicting Power Plant Output
**Notebook:** `notebooks/04_tuning_features.ipynb`

Last week the random forest won by about 3.5 MW of error, and gradient boosting came second.
This week you'll try to make your models even better in 2 ways
1. adjusting their settings : **tuning**
2. giving them new inouts : **feature engineering**

**By the end of this week you will be able to:**

- Explain what a hyperparameter is and tune one with cross-validation
- Create new features from existing ones and test whether they help
- Lock in a final model without touching the test set

Plan ~ 2 hours (some cells may take more than a minute to run)

---

## Step 1: Hyperparameters

A model's **parameters** are learned from data (like a regression coefficient). Its **hyperparameters** are settings you choose before training (like the number of trees in a forest or how deep each tree can grow).

Some common ones for a random forest:

| Hyperparameter | What it controls |
|---|---|
| `n_estimators` | Number of trees |
| `max_depth` | How deep each tree can grow (deeper means more detail, and more risk of overfitting) |
| `min_samples_leaf` | Minimum rows in each final leaf (larger means smoother predictions) |
| `max_features` | How many features each split can consider |
| `learning_rate` | How big a step each new tree takes to fix earlier mistakes |

For gradient boosting, look at `n_estimators`, `learning_rate`, and `max_depth`.

---

## Step 2: Tune Random Forest with Cross-Validation

![random-forest](random-forest.jpg)

- Open `notebook` folder
- Click **New → Python 3 (ipykernel)**
- Name notebook `04_tuning_features`

`GridSearchCV` tries **every combinations** of your settings list, scores each one with cross-validation and keeps the best.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv("../data/raw/power_plant_train.csv")
X_train = df[["AT", "V", "AP", "RH"]]
y_train = df["PE"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
}

rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
)
rf_search.fit(X_train, y_train)

print("Best settings:", rf_search.best_params_)
print("Best CV RMSE:", -rf_search.best_score_)
```

The grid has 2 x 3 x 3 = 18 combinations and each one is trained 5 times (once per fold). 
18 x 5 = 90 forests! It takes about 1-3 minutes so better to keep grid small and adding up after (that way your computer doesn't crash)

To see full results:

```python
rf_results = pd.DataFrame(rf_search.cv_results_)
rf_results["RMSE"] = -rf_results["mean_test_score"]
rf_results.sort_values("RMSE")[["params", "RMSE", "std_test_score"]].head()
```

---

## Step 3: Tune Gradient Boosting

Gradient boosting came second last week but it was using default settings. Try this.

```python
gb_grid = {
    "n_estimators": [200, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
}

gb_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
gb_search.fit(X_train, y_train)

print("Best settings:", gb_search.best_params_)
print("Best CV RMSE:", -gb_search.best_score_)
```

Answer in Markdown cell:

1. How much did tuning improve each model's CV RMSE, compared with week03 (random forest ~ 3.50 , gradient boosting ~ 3.91)?
2. Which tuned model is best now? Is that a surprise after last week?
3. Did any best setting land at the **edge** of its grid (the biggest or smallest value you tried)?
4. If so, the true best might be outside the grid. Try one value further and see.


---

## Step 4: Feature Engineering

**Feature engineering** means creating new columns from the ones you have. Try a few ideas based on what you learned in Week 01:

```python
def add_features(data):
    out = data.copy()
    out["AT_squared"] = out["AT"] ** 2      # the AT-PE plot bent slightly
    out["AT_x_V"] = out["AT"] * out["V"]    # AT and V move together
    out["AT_x_RH"] = out["AT"] * out["RH"]  # humidity may matter more on hot days
    return out

X_train_fe = add_features(X_train)
```

Other ideas: ratios (`AT / V`), logs, or interactions between any two columns. Pick ideas that have a **reason** (for example, "the AT-PE plot was curved, so AT squared might help").

**Test each idea fairly:**

```python
from sklearn.linear_model import LinearRegression

for label, data in {"original": X_train, "with new features": X_train_fe}.items():
    scores = cross_val_score(LinearRegression(), data, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
    print(label, round(-scores.mean(), 3), "+/-", round(scores.std(), 3))
```

**Your turn** : run the same comparison with your tuned random forest and your tuned gradient boosting.
To make a fresh model with the best settings, use **, which unpacks a dictionary into settings:

```python
best_gb = GradientBoostingRegressor(random_state=42, **gb_search.best_params_)
```

Then also try adding the new features **one at a time** to linear regression, to see which one does the most work.

**Questions:**

1. Did the new features help linear regression? By how much? Which single feature helped most?
2. Did they help the tree models? Why might trees gain less from hand-made features than a linear model does?
3. Did any idea make things worse or change nothing? That's a result worth reporting too.

> Keep your `add_features` function safe. In Week 05 you'll paste the **identical** function into that notebook to process the test set. Don't evaluate on the test set yet.

---

## Step 5: Lock Your Final Model

Decide on one final model: a specific algorithm, its tuned settings, and its feature set. Write it down in your notebook in a Markdown cell:

> **Final model:** Random forest, `n_estimators=...`, `max_depth=...`, using features `...`. Chosen because ...

Then fit it on the **full training set** and save it, so you don't need to retrain every time:

```python
import joblib

final_model = GradientBoostingRegressor(random_state=42, **gb_search.best_params_)   # use your own choice
final_model.fit(X_train, y_train)    # use X_train_fe if your final model uses the new features

joblib.dump(final_model, "../results/final_model.joblib")
```
Two things to know about saved models:

- Only load a  `.joblib` file you made yourself or trust, because loading one can run code on your computer.
- A model saved with one version of scikit-learn may not load in another, so in Week 06 you'll save your exact package versions.

Don't touch the test file until Week 05. If you change your mind after seeing test results, the test file stops being honest.

---

## Deliverable

`04_tuning_features.ipynb`, containing:

- [ ] A tuning search with a small, clearly stated grid
- [ ] Best settings and the cross-validated improvement over the defaults
- [ ] At least 3 engineered features, each with a one-line reason
- [ ] A before-and-after comparison table for the features
- [ ] A Markdown cell stating your final model and why you chose it
- [ ] The saved model in `results/final_model.joblib`
- [ ] A short written answer (5-8 sentences): What helped, what didn't, and what surprised you?

---

## Common Mistakes

- **Trying lots of changes and keeping the luckiest.** If a gain is smaller than the fold-to-fold spread, it may be noise.
- **Engineering features with no reason.** Every feature should have a story you can tell.
- **Making the grid huge.** More combinations means slower runs and more chances for lucky results.

---

## Check Your Answers

- **Random forest** tuning barely helps. Best settings: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=1`, CV RMSE ≈ 3.49, against ≈ 3.50 with the defaults. The defaults were already good. Notice that two settings landed at the edge of the grid (300 trees is the most you tried, and 1 is the smallest leaf).
- **Gradient boosting** tuning helps a lot. Best settings: `n_estimators=500`, `learning_rate=0.1`, `max_depth=5`, CV RMSE ≈ 3.23. It went from second place (3.91) to first, beating the forest by about 0.26 MW. The std is about 0.15, so this is a real win.
- Its best settings are also at the edge of the grid. Going one step further (1,000 trees) gives about 3.18. It's a small extra gain, and it takes twice as long to train.
- **New features help linear regression**: CV RMSE goes from ≈ 4.56 to ≈ 4.33. AT_squared alone does most of it (≈ 4.34), which fits the slight bend you saw in Week 01.
- **New features** don't help the trees: the forest goes from ≈ 3.49 to ≈ 3.53, and gradient boosting stays at ≈ 3.24. Trees can already find bends and interactions by themselves, so extra columns just add noise.
- A reasonable final model: tuned gradient boosting on the original 4 columns.
The big lesson: always tune your top two or three models, not just the winner. Default settings can hide how good a model really is.

---

## Next Week

In Week 05, you'll open the test file, score your final model once, and dig into where and why it makes mistakes.
