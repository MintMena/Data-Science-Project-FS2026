# Week 05: Tune the Best Models and Combine Them

**Project:** Predicting House Prices
**Notebook:** `notebooks/05_tuning.ipynb`

This week you improve your best models by adjusting their settings, then combine them into one stronger model.

**By the end of this week you will be able to:**

- Explain what hyperparameters are
- Use `RandomizedSearchCV` to tune a model
- Build an ensemble that averages several models

Plan for about 1.5 to 2 hours. Some cells take up to a minute to run.

---

## New Idea: Hyperparameters

Hyperparameters are settings you choose before training, such as the number of trees in a forest or how deep each tree can grow. Tuning means trying many combinations and keeping the one with the best CV score.

---

## Step 1: Tune Random Forest

- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `05_tuning`

Copy your setup cell from Week 04 (the one with `cv`) to the top again. Then:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 8, 12, 16],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": [0.4, 0.6, 0.8, 1.0],
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), grid,
    n_iter=40, cv=cv, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)

print(search.best_params_, -search.best_score_)
best_rf = search.best_estimator_
```

`n_iter=40` means it tries 40 random combinations from the grid, instead of all 192.

---

## Step 2: Your Turn - Tune Three More

Tune Gradient Boosting, XGBoost and SVR the same way. Suggested ranges:

| Setting | Try |
|---|---|
| Number of trees | 200 to 800 |
| Learning rate | 0.01 to 0.1 |
| Max depth | 2 to 6 |
| Subsample | 0.6 to 1.0 |
| SVR C | 5 to 200 |
| SVR epsilon | 0.1 to 2 |

SVR sits inside a pipeline, so its setting names start with `svr__`, for example `svr__C`.

> **Hint**
> 1. For Gradient Boosting and XGBoost, one grid works for both: `{"n_estimators": [200, 400, 600, 800], "learning_rate": [0.01, 0.03, 0.05, 0.1], "max_depth": [2, 3, 4, 5, 6], "subsample": [0.6, 0.8, 1.0]}`
> 2. For SVR, use `make_pipeline(StandardScaler(), SVR())` with `{"svr__C": [5, 10, 20, 50, 100, 200], "svr__epsilon": [0.1, 0.5, 1, 1.5, 2], "svr__gamma": ["scale", 0.05, 0.1, 0.2]}`
> 3. Give each model `random_state=42` (except SVR) and keep `n_iter=40`, `cv=cv`, `random_state=42` in the search
> 4. Save each winner as `best_gb`, `best_xgb` and `best_svr`

---

## Step 3: Combine Them

An **ensemble** averages several models' predictions. Different models make different mistakes, so averaging can cancel some of them out.

```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([("rf", best_rf), ("gb", best_gb),
                            ("xgb", best_xgb), ("svr", best_svr)])
```

Score the four tuned models and the ensemble with the same `cv` and validation set as Week 04, and add them to your results table. (You can reuse your Week 04 loop: put the five models in a new dictionary and run it.)

**Questions to answer in writing:**

1. How much did tuning improve each model's CV RMSE, compared with Week 04?
2. Did the ensemble beat every single model? By more than the CV std?
3. Which model will you choose as your final model, and why?

---

## Step 4: Check In

When you finish, check in so we know you're still with us. It takes 1 minute (you need a free GitHub account):

1. Open the [check-in form](https://github.com/MintMena/Data-Science-Project-FS2026/issues/new?template=checkin.yml)
2. Pick **Real Estate** and **Week 05**, and type your name (a first name is fine, it shows publicly)
3. Say whether you'll join the model contest (the date is coming soon), then click **Create**

A bot replies to confirm.

---

## Deliverable

`05_tuning.ipynb`, containing:

- [ ] At least 3 models tuned, with their best settings printed
- [ ] An ensemble built and scored
- [ ] A results table with CV RMSE, CV std and validation RMSE
- [ ] Final model chosen by CV score, with a written reason
- [ ] Your Week 05 check-in

---

## Common Mistakes

- **Huge search grids.** They take forever. Start small and widen only if needed.
- **Choosing by validation score.** Use CV RMSE, like Week 04.
- **Looking for test file.** Ignore it until Week 06.

---

## Answers

- Tuned models (CV RMSE):
  - Random Forest ≈ 6.72 (`n_estimators=800`, `max_depth=12`, `min_samples_leaf=2`, `max_features=0.6`)
  - XGBoost ≈ 6.81 and Gradient Boosting ≈ 6.83 (both `n_estimators=400`, `learning_rate=0.01`, `max_depth=3`, `subsample=0.6`)
  - SVR ≈ 7.01 (`C=20`, `epsilon=1.5`, `gamma=0.05`)
- Tuning helps the boosting models (≈ 7.0-7.1 → ≈ 6.8) and SVR (≈ 7.45 → ≈ 7.01). Random Forest moves only a little (≈ 6.86 → ≈ 6.72).
- Ensemble: CV RMSE ≈ 6.71, validation RMSE ≈ 6.49. It ties with the best single models (untuned Extra Trees was ≈ 6.70). Every gap is far smaller than the CV std (≈ 1).
- A fair final choice: the ensemble. It's as good as the best model and less likely to have one bad day, because it averages four. Any choice is fine if your reason uses CV RMSE.

---

## Next Week

In Week 06, you'll test your final model, write your conclusions, and get ready for the model contest.
