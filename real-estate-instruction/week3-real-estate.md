# Week 03: Split the Data and Build Your First Model

**Project:** Predicting House Prices
**Notebook:** `notebooks/03_baseline.ipynb`

Last week you built new features and saved them. 

This week you will train your first model : **linear regression**, the simplest one there is. It will become your **baseline**, the score every later model has to beat. 

**By the end of this week you will be able to:**

- Remove the outlier the right way and explain why
- Split data into training and validation sets
- Train a model and make predictions
- Measure error with RMSE, MAE and R²

Plan for about 1 to 1.5 hours

---

## New Idea : Training, Validation and Test Data

- **Training Data** : what you train your model and what it will learn from. 
- **Validation Data** : Use to check on how you're doing and to comapre models while you build, a 'mock' test data.
- **Test Data** : The actual testing you'll use on model, something model has not seen yet.

Let's say you have final exam coming up; training data is your study materials, validation data is your practice exam, you will be prepare for real deal which is final exam, test data.

Your `real_estate_train.csv` is all study materials.
This week you will cut a practice exam out of it.

---

## Step 1 : Load the Data and Remove Outlier

Do the usual step
1. Open `notebook` folder
2. Click **New → Python 3 (ipykernel)**
3. Name the notebook `03_baseline`
4. Load the features you saved last week

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/houses_features.csv")
print(df.shape)
```
Last week you found one house priced at 117.5 which is way above every similar house. A model would try to bend itself to fit that one strange sale and get worse predictions for normal houses. 
That's why we need to remove it now before training.

```python
upper = 103.6   # the cutoff from Week 02 (use your exact value)
df = df[df["price"] <= upper]
print(df.shape)
```

We only remove it from the training data. 
The test file never had it (we kept outliers out on purpose).

---

## Step 2 : Choose the Features and Split

```python

# short way to make a list: every column name, except price and trand_date

FEATURES = [c for c in df.columns if c not in ("price", "trans_date")]
print(FEATURES)

X = df[FEATURES]
y = df["price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_val.shape)
```
We leave out `trans_date` because `trans_month` holds the exact same information. Two columns that say same thing will confuse linear regression since it can't decide how to split the credit between them, and its number come out huge and meaningless.
- `test_size = 0.2` puts 20% of rows in validation set
- `random_state = 42` makes the split the same every time. You'll copy this exact cell into later notebooks, so every week uses the same split.

---

## Step 3 : A Dummy model

Before a real model, we will make one that has no skill at all.
It will always predict the average price from training data.

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
```
Any real model should beat this very easily. If it doesn't, something is wrong.

---

## Step 4: Train Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)     # learn from the training data
pred = model.predict(X_val)     # predict the validation houses
```
- `.fit()` is where the model learns (only sees `X_train` and `y_train`)
- `.predict()` gives a predicted price for each validation house.

---

## Step 5 : Measure the Error

| Metric | Meaning | Better when |
|---|---|---|
| **RMSE** (root mean squared error) | typical size of an error, in price units. Big misses count extra. | lower |
| **MAE** (mean absolute error) | average size of an error, in price units | lower |
| **R²** | share of the ups and downs in price that the model explains (0 to 1) | higher |

Since you'll score more than one model, put the scoring in a function:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
    }

results = pd.DataFrame({
    "dummy": evaluate(dummy, X_val, y_val),
    "linear regression": evaluate(model, X_val, y_val),
}).T
results
```
Remember the unit: a price of 40 means 400,000 NTD per Ping. So an RMSE of 7 means predictions are typically off by about 70,000 NTD per Ping.

Finally, plot predicted vs actual prices for the validation houses:

```python
plt.scatter(y_val, pred)
plt.plot([0, 80], [0, 80], "r--")  # the line of perfect predictions
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("Linear regression on validation set")
plt.savefig("../reports/figures/linear_pred_vs_actual.png")
plt.show()
```
Answer in a Markdown cell:

1. How many rows are in the training and validation sets?
2. What does your RMSE mean in plain words?
3. How much better is linear regression than the dummy?
4. Why did we remove the outlier, and why only from the training data?
5. Why do we need a validation set, if we already have a test file coming up?

---

## Deliverable

`03_baseline.ipynb`, containing:

- [ ] The outlier removed from the training data, with a sentence explaining why
- [ ] The feature list (without `price` and `trans_date`), with a sentence on why `trans_date` is left out
- [ ] A training/validation split with `random_state=42`
- [ ] The dummy and linear regression trained, and scored on the validation set with RMSE, MAE and R²
- [ ] A predicted-vs-actual plot saved to `reports/figures/`

---

## Check Your Answers

- 290 rows before removing the outlier, 289 after.
- 8 features: `house_age`, `n_stores`, `lat`, `lon, dist_center_km`, `log_dist_mrt`, `trans_month`, `age_sq`.
- Training: 231 houses. Validation: 58 houses.
- Dummy: RMSE ≈ 14.7, MAE ≈ 12.1, R² ≈ 0.
- Linear regression: RMSE ≈ 7.3, MAE ≈ 4.9, R² ≈ 0.75. That's about half the dummy's error. In plain words, predictions are typically off by about 7 units, or about 70,000 NTD per Ping.
- RMSE is quite a bit bigger than MAE. That means a few houses have big errors (one validation house is off by about 31) while most are much closer. About 2 out of 3 validation houses are within 5 units.

---

## Common Mistakes

- **Fitting on the validation set.** `.fit()` should only every see `X_train` and `y_train`
- **Forgetting `random_state`** Without it you get a different split every run, and your numbers won't match anyone else's (or your own next week).
- **Keeping both trans_date and trans_month.** The predictions still work, but the model's numbers become meaningless.
- **Forgetting Unites** An RMSE of 7.3 means 7.3 × 10,000 NTD per Ping, not 7.3%.

---

## Next Week

In Week 04, you'll compare about ten different models and learn cross-validation, a fairer way to score them than one split.
