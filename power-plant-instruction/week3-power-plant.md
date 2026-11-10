# Week 03: Compare Models with Cross-Validation

**Project:** Predicting Power Plant Output
**Notebook:** `notebooks/03_model_comparison.ipynb`

Linear regression gave you a baseline. This week you try more flexible models and compare them fairly. A single train/test score can be lucky or unlucky, so you'll use cross-validation instead.

**By the end of this week you will be able to:**

- Explain what cross-validation does and why it's more reliable than one split
- Compare several regression models with the same fair procedure
- Decide whether a more complex model is worth its extra cost

Plan for about 1.5 hours

---

## Step 1: How Cross-Validation Works

In **5-fold cross-validation**, the training data is split into 5 equal parts. The model is trained 5 times, each time holding out a different part for scoring. You get 5 scores instead of 1, so you can see both the average **and** how much it varies.

You'll still use only the **training set**. The test set stays untouched until Week 05.

Start the notebook with the same split as before:
- Open the notebook folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `03_model_comparison`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("../data/raw/power_plant_train.csv")
X_train = df[["AT", "V", "AP", "RH"]]
y_train = df["PE"]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
```
- `shuffle=True` mixes up the rows before cutting them into folds.
- `random_state=42` makes the folds the same every time you run it, for every model. That's what makes the comparison fair.

---

## Step 2: The Models

| Model | Idea in one sentence |
|---|---|
| **Linear regression** | A straight-line formula (your baseline from Week 02) |
| **Decision tree** | A flowchart of yes/no questions about the features |
| **Random forest** | Many decision trees, each trained on a random sample, then averaged |
| **Gradient boosting** | Trees built one after another, each fixing the previous ones' mistakes |
| **k-nearest neighbors** | Predicts from the most similar rows in the training data (needs scaling) |

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

models = {
    "linear": LinearRegression(),
    "decision tree": DecisionTreeRegressor(random_state=42),
    "random forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient boosting": GradientBoostingRegressor(random_state=42),
    "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor()),
}
```

Notice kNN is wrapped in a pipeline with a scaler, because distance-based models are sensitive to feature scale.

---

## Step 3: Score Every Model the Same Way

```python
rows = []
for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["neg_root_mean_squared_error", "r2"], n_jobs=-1
    )
    rmse = -scores["test_neg_root_mean_squared_error"]   # flip the sign (see below)
    rows.append({
        "model": name,
        "RMSE mean": rmse.mean(),
        "RMSE std": rmse.std(),
        "R2 mean": scores["test_r2"].mean(),
        "fit time (s)": scores["fit_time"].mean(),
    })

comparison = pd.DataFrame(rows).sort_values("RMSE mean")
comparison
```

> scikit-learn reports error metrics as **negative** numbers so that "higher is better" holds for all scorers. Flip the sign to read them as normal errors.

- **std** (standard deviation) tells you how much the 5 fold scores jump around. Small std = steady model.
- The random forest takes the longest, around 10 to 20 seconds. That's normal.

Do the same with `scoring="r2"` to get R², and combine both in one table.

---

## Step 4: Plot the Comparison

Make a bar chart or box plot of RMSE across models, with the spread (standard deviation) visible:

```python
comparison.plot(x="model", y="RMSE mean", kind="bar", yerr="RMSE std", legend=False)
plt.ylabel("cross-validated RMSE (MW)")
plt.title("Model comparison") #the lower the better
plt.tight_layout()
plt.savefig("../reports/figures/model_comparison.png")
plt.show()
```

---

## Step 5: Think About Whether It's Worth It

Look at your table and answer in writing:

1. Which model has the lowest average RMSE? By how much does it beat linear regression, in MW?
2. Is that improvement big compared with the spread between folds (the std)? If the gap between two models is smaller than their std, they may not really be different.
3. Which models are the easiest to explain to someone who doesn't code? Which are the slowest to train? Is the accuracy gain worth those costs?
4. Why might a single decision tree do worse than a random forest, even though the forest is made of trees?

The most useful lesson this week may be that the best model isn't always much better than a simple one.

---

## Deliverable

`03_model_comparison.ipynb`, containing:

- [ ] Cross-validation set up with a fixed seed (`shuffle=True` and `random_state=42`)
- [ ] All 5 models compared, including your linear baseline
- [ ] A table of mean and standard deviation of RMSE (and R²) for each model
- [ ] A plot comparing the models, saved to `reports/figures/`
- [ ] A short written answer (5-8 sentences) to the questions in Step 5
- [ ] A note saying which one or two models you'll carry into Week 04

---

## Common Mistakes

- **Comparing models on different splits.** Use the same `cv` object for every model.
- **Only reading the average.** Always look at the std too. A small win inside the spread isn't a real win.
- **Forgetting the scaler on kNN.** Without it, features with large numbers dominate the distance.
- **Leaving out random seeds (`random_state`).** Without them, your numbers change every run and can't be reproduced.

---

## Check Your Answers

Only look after you've tried the steps yourself. Small differences are fine.

| Model | RMSE mean (MW) | RMSE std | R² |
|---|---|---|---|
| random forest | ≈ 3.50 | ≈ 0.15 | ≈ 0.958 |
| gradient boosting | ≈ 3.91 | ≈ 0.14 | ≈ 0.948 |
| kNN | ≈ 3.94 | ≈ 0.15 | ≈ 0.947 |
| linear | ≈ 4.56 | ≈ 0.14 | ≈ 0.929 |
| decision tree | ≈ 4.77 | ≈ 0.12 | ≈ 0.922 |

- The random forest wins, about 1.1 MW better than linear regression. That gap is around 7 times bigger than the std, so it's a real difference, not luck.
- Gradient boosting and kNN are basically tied. Their gap (0.03 MW) is much smaller than the std.
- A single decision tree is the **worst** model, even worse than the straight line. One tree memorizes the training rows too closely (this is called **overfitting**). A forest averages many different trees, so their individual mistakes cancel out.
- The forest is the slowest to train by far, and the hardest to explain. Whether 1 MW is worth it depends on who uses the model: linear regression already explains about 93% of the ups and downs.

---

## Next Week

In Week 04, you'll tune your best model's settings, try a few new features, and lock in your final model.
