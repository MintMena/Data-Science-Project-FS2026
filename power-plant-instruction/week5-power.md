# Week 05: Evaluate and Interpret

**Project:** Predicting Power Plant Output
**Notebook:** `notebooks/05_interpretation.ipynb`

This is the week you open the test file. You'll score your final model once, then look past the single number to understand what the model learned and where it fails.

**By the end of this week you will be able to:**

- Evaluate a final model on held-out data
- Read residual plots and spot patterns in the errors
- Explain which features matter most, and how you know
- Describe the model's limits honestly

Plan for about 1.5 hours.

---

## Step 1: The One Test Evaluation

1. Download `power_plant_test.csv` from the club GitHub (same place as Week 01)
   https://github.com/MintMena/Data-Science-Project-FS2026
2. Move the file into `data/raw/`

- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `05_interpretation`

Load both files and your saved model:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train = pd.read_csv("../data/raw/power_plant_train.csv")
test = pd.read_csv("../data/raw/power_plant_test.csv")
X_train, y_train = train[["AT", "V", "AP", "RH"]], train["PE"]
X_test, y_test = test[["AT", "V", "AP", "RH"]], test["PE"]

model = joblib.load("../results/final_model.joblib")
```

If your final model uses engineered features, paste the same `add_features` function from Week 04 at the top of this notebook, exactly as it was, and apply it to `X_test` first.

```python
pred = model.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("Test MAE: ", mean_absolute_error(y_test, pred))
print("Test R2:  ", r2_score(y_test, pred))
```

Do this **once**. Then compare:

1. How does the test RMSE compare with the cross-validated RMSE from Week 04? A big gap would suggest overfitting.
2. Compare it with the linear baseline's test RMSE (fit `LinearRegression` on `X_train`, `y_train` and score it on the test file). How much did all your work improve things, in MW?

---

## Step 2: Residuals

A **residual** is the actual value minus the prediction. If the model is good, residuals should look like random noise with no pattern.

```python
residuals = y_test - pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(pred, residuals, s=3, alpha=0.3)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted PE")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs predicted")

axes[1].hist(residuals, bins=40)
axes[1].set_title("Distribution of residuals")

axes[2].scatter(y_test, pred, s=3, alpha=0.3)
lims = [y_test.min(), y_test.max()]
axes[2].plot(lims, lims, "r--")
axes[2].set_xlabel("Actual PE")
axes[2].set_ylabel("Predicted PE")
axes[2].set_title("Predicted vs actual")

plt.tight_layout()
plt.savefig("../reports/figures/residuals.png")
plt.show()
```

**How to read them:**

- **Residuals vs predicted:** you want a flat, even cloud around zero. A curve or a fan shape means the model is missing something.
- **Histogram:** roughly bell-shaped and centered at zero is good. A long tail means a few big misses.
- **Predicted vs actual:** points should hug the dashed line.

**Questions to answer in writing:**

1. Is the average residual close to zero?
2. Do errors get larger for high or low power outputs?
3. Is there any visible pattern, or does it look like random noise?

---

## Step 3: Where Does the Model Fail?

Find the worst predictions and look at them:

```python
errors = X_test.copy()
errors["actual"] = y_test
errors["predicted"] = pred
errors["abs_error"] = (y_test - pred).abs()

errors.sort_values("abs_error", ascending=False).head(10)
```

Also check error by range. For example, split `AT` into four bands and compare the average error in each:

```python
errors["AT_band"] = pd.qcut(errors["AT"], 4, labels=["coldest", "cool", "warm", "hottest"])
errors.groupby("AT_band", observed=True)["abs_error"].mean()
```

**Questions:**

1. What do the worst predictions have in common (unusual temperature, humidity, extreme output)?
2. Is the model better in some conditions than others? What might explain that?

---

## Step 4: Which Features Matter?

**Permutation importance** shuffles one feature at a time and measures how much the model's error increases. A big increase means the model relies on that feature.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)

importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
importance.plot(kind="barh")
plt.xlabel("Increase in RMSE when feature is shuffled")
plt.savefig("../reports/figures/feature_importance.png")
plt.show()
```

Compare this with your Week 01 correlation heatmap:

1. Does the most important feature match the one with the strongest correlation?
2. Two features (`AT` and `V`) are strongly correlated with each other. How might that affect importance scores? (When two features carry similar information, shuffling one hurts less, because the model leans on the other.)

Optional: make partial dependence plots to see how predictions change as one feature changes:

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_test, ["AT", "V", "AP", "RH"])
plt.show()
```

---

## Step 5: What Can't This Model Tell You?

Write a short, honest list of limits. Some prompts:

- The data comes from one plant over about six years. Would the model work at a different plant?
- The model predicts from four measurements only. What else would affect output?
- How large is the typical error compared with the range of `PE`? Is that good enough for planning grid supply?
- What would happen if conditions fell outside the range of the training data?

---

## Step 6: Check In

When you finish, check in so we know you're still with us. It takes 1 minute (you need a free GitHub account):

1. Open the [check-in form](https://github.com/MintMena/Data-Science-Project-FS2026/issues/new?template=checkin.yml)
2. Pick **Power Plant** and **Week 05**, and type your name (a first name is fine, it shows publicly)
3. Say whether you'll join the model contest (the date is coming soon), then click **Create**

A bot replies to confirm.

---

## Deliverable

`05_interpretation.ipynb`, containing:

- [ ] A single test evaluation, compared with cross-validation and with the linear baseline
- [ ] Residual plots with a written reading of them
- [ ] A look at the worst predictions and error by range
- [ ] A permutation importance chart, with a comparison to the correlations
- [ ] A written "limits of this model" section (6-10 sentences)
- [ ] Your Week 05 check-in

---

## Common Mistakes

- **Evaluating on the test file more than once.** After that, you're tuning to it.
- **Showing only R².** A high R² can hide large errors in specific conditions.
- **Treating importance as cause.** Importance says what the model uses, not what causes power output.
- **Skipping the limits.** An honest list of limits makes your project stronger, not weaker.

---

## Answers

These are for tuned gradient boosting (`n_estimators=500`, `learning_rate=0.1`, `max_depth=5`) on the 4 original columns.

- Test file: 1,914 readings.
- Test RMSE ≈ 3.11 MW, MAE ≈ 2.19 MW, R² ≈ 0.966. The CV RMSE was ≈ 3.23, so the test score is just as good. No overfitting.
- Linear baseline on the test file: RMSE ≈ 4.58 MW. Your work cut the error by about 1.5 MW (about 32%).
- Average residual ≈ 0, and about 93% of predictions are within 5 MW.
- The histogram has a long tail on one side: a few hours where the plant made **much less** power than the weather suggests. The worst one is a cold hour (6.2 °C) with 443.7 MW actual vs 487.2 MW predicted, a 43.5 MW miss. The plant was probably not running normally that hour, which the four measurements can't show.
- Error is about the same in every temperature band (≈ 2.1-2.3 MW).
- Importance: `AT` ≫ `V` > `AP` ≈ `RH` (RMSE goes up by about 16.8, 3.8, 0.9 and 0.7 MW). `V` looks less important than its correlation suggested, because `AT` carries much of the same information.

---

## Next Week

In Week 06, you'll write up the project, package it so someone else can reproduce your results, and get ready for the model contest.
