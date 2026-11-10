# Week 03: Prepare the Data and Build Your First Model

**Project:** Predicting Heart Failure Survival
**Notebook:** `notebooks/03_baseline.ipynb`

Last week we've explored patients and learned why `time` is off-limits. Now it's time for your first model.

This week you will prepare the features and train **logistic regression** as your **baseline** : the score every model has to beat later on.

**By the end of this week you will be able to:**

- Explain class imbalance and why accuracy can fool you
- Explain why the test set is kept separate from everything else
- Score a model with repeated cross-validation
- Read MCC, recall, ROC-AUC and a confusion matrix

Plan for about 1.5 to 2 hours

---

## New Idea : Imbalance and Better Metrics

Only about 32% of patients in our file died. A lazy model that always says survived would be about 68% accurate and still completely useless because it never catches a single death.

So accuracy is not our main score. We'll use these instead.

| Metric | What it means | Range |
|---|---|---|
| **MCC** (Matthews correlation coefficient) | overall quality, fair even when one class is much bigger. **Your main score.** | −1 to 1 (0 = no better than guessing) |
| **Recall** | of the patients who died, what share did the model catch? | 0 to 1 |
| **ROC-AUC** | how well the model ranks risky patients above safe ones | 0.5 (random) to 1 (perfect) |

---

## Step 1 : Prepare the Features

Do the usual step
1. Open `notebook` folder
2. Click **New → Python 3 (ipykernel)**
3. Name the notebook `03_baseline`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/raw/heart_failure_train.csv")
```
Let's keep things organized. First, we'll use every column except `time` to create feature list

1. **Make feature list**
Use every column except `time` (leakage!) and `DEATH_EVENT`

```python
FEATURES = [c for c in data.columns if c not in ("time", "DEATH_EVENT")]
print(len(FEATURES), FEATURES)
```
2. **Fix the lopsides columns**
Three lab values have a few huge numbers that stretch everything out. Taking the logarithm squeezes big values together and evens things out:

```python
for col in ["creatinine_phosphokinase", "serum_creatinine", "platelets"]:
    data[col] = np.log(data[col])
```
**Your turn** : re-draw the histograms of those 3 columns (like Week 01) and compare them with before. Which changed the most?

3. **Save the prepared table**

```python
data.to_csv("../data/processed/patients.csv", index=False)
```
This keeps `time` and `DEATH_EVENT` in the file. Since `FEATURES` never includes `time`. Write down the 3 log steps somewhere safe : in week 06 , you'll do the **exact same** steps to the test file.

---

## Step 2 : Set Up the Training Data

```python
X_train = data[FEATURES]
y_train = data["DEATH_EVENT"]
print(X_train.shape, y_train.mean())
```
You don't need to split anything. Your whole file is training data. 
We already set aside a test file (60 patients) that you'll open once, in Week 06. We also **stratified** it, which means the test file has the same 32% death rate as yours. 
Without that, a small test set could end up with too few deaths to score fairly.

> **Rule from now** : every choice you make (features, models, settings) is based only on the training data.

---

## Step 3 : Train Logistic Regression

Despite the name, **logistic regression** is a **classification model**. 
It gives each patient a probability of dying, from 0 to 1.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=2000))
```
- `StandardScaler()` puts all the features on the same scale, so age (around 60) doesn't get treated differently from creatinine (around 0 after the log).
- `class_weight="balanced"` tells the model to pay extra attention to the smaller "died" group. Without it, the model would lean toward predicting "survived" for almost everyone.
- `make_pipeline` glues the two steps together, so the scaling is always learned from the right rows.

---

## Step 4 : Score with Repeated Cross-Validation

**Cross-validation** cuts the training data into 5 parts (folds), trains on 4 and scores on the 5th, so every part gets a turn. With only 209 patients, the score jumps around a lot depending on how the folds are cut. So we repeat the whole thing 10 times with different shuffles and average all 50 scores:

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {"MCC": make_scorer(matthews_corrcoef), "AUC": "roc_auc", "Recall": "recall"}

scores = cross_validate(model, X_train, y_train, cv=rcv, scoring=scoring)
for name in scoring:
    print(name, round(scores["test_" + name].mean(), 3), "+/-", round(scores["test_" + name].std(), 3))
```
- **Stratified** means every fold keeps the same death rate.
- `rcv` and `scoring` will be copied into every later notebook, so every model is scored the exact same way.

---

## Step 5 : Confusion Matrix

A **confusion matrix** is a 2 × 2 grid of right and wrong predictions:

| | predicted survived | predicted died |
|---|---|---|
| **really survived** | correct | false alarm |
| **really died** | **missed death** | correct |

To fill it fairly, every patient needs a prediction from a model that did **not** train on them. `cross_val_predict` does that for you. It needs plain (not repeated) folds:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X_train, y_train, cv=cv5)

ConfusionMatrixDisplay.from_predictions(y_train, predictions,
                                        display_labels=["survived", "died"])
plt.savefig("../reports/figures/baseline_confusion_matrix.png")
plt.show()
```

Answer in a Markdown cell:

1. Why is accuracy a poor main metric for this data?
2. What does recall mean in medical terms? Why might a doctor care about it the most?
3. Read your confusion matrix: how many deaths did the model miss? How many false alarms did it raise?
4. Which log-transformed column changed the most?

---

## Deliverable

`03_baseline.ipynb`, containing:

- [ ] A feature list of 11 columns, without `time` and `DEATH_EVENT`
- [ ] The three log-transformed columns, with before-and-after histograms
- [ ] The prepared table saved to `data/processed/patients.csv`
- [ ] Logistic regression scored with repeated cross-validation (MCC, ROC-AUC and recall, with their std)
- [ ] A confusion matrix, saved to `reports/figures/`

---

## Check Your Answers

- 11 features. Training data: 209 patients, 68 deaths (about 32.5%). Test file (Week 06): 60 patients, 19 deaths.
- The log transform evens out `creatinine_phosphokinase` the most (from very lopsided to almost balanced). `serum_creatinine` gets much better but still has a tail. `platelets` was only a little lopsided, and now it leans slightly the other way.
- Logistic regression with repeated CV: MCC ≈ 0.35, ROC-AUC ≈ 0.75, recall ≈ 0.65.
- The scores jump around a lot between folds (MCC std ≈ 0.13). With this few patients, that's normal.
- Confusion matrix: the model catches 47 of the 68 deaths and **misses 21**. It also raises 41 false alarms (survivors flagged as "died").
- Its accuracy is about 70%, barely above the 67.5% you'd get by always saying "survived". But the lazy model catches zero deaths, while this one catches about 2 out of 3. That's exactly why we don't use accuracy as the main score.
---

## Common Mistakes

- **Including `time` in the feature list.** `Always check `FEATURES` before training.
- **Reporting only accuracy.** A useless model gets 67.5%. Lead with MCC and recall.
- **Using `rcv` in `cross_val_predict`** It needs plain folds (`cv5`), not repeated ones, or it will give an error.
- **Forgetting log steps later** Whatever you do to the training data here, you must do to the test file in Week 06.
---

## Next Week

In Week 04, you'll compare about eleven classifiers and find out which measurements matter most.
