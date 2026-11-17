# Week 04: Handle Imbalance and Compare Models
 
**Project:** Classifying Poker Hands
**Notebook:** `notebooks/04_imbalance_models.ipynb`
 
Even with good features, the rare hands have very few examples. This week you try techniques for imbalanced data and compare more models, and you learn the leakage mistake that makes imbalance fixes look better than they are.
 
**By the end of this week you will be able to:**
 
- Use class weights and oversampling for imbalanced data
- Explain why oversampling must happen **after** splitting, and inside cross-validation
- Compare models with stratified cross-validation
- Choose a final model and feature set
- 
Plan for about ~ 2 hours.
 
---

![poker-fold](images/poker-fold.jpg)
 
## Step 1: Set Up
 
- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `04_imbalance_models`
- Make sure `features.py` from Week 03 is in the same folder
Use your engineered features from Week 03 and the same split:
 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from features import make_features
 
cols = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "hand"]
data = pd.read_csv("../data/raw/poker_train.csv", header=None, names=cols)
X = data.drop(columns="hand")
y = data["hand"]
 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_fe = make_features(X_train)
X_val_fe = make_features(X_val)
```
 
Because the rarest classes have so few examples, a single validation set is shaky. This week you'll also use **stratified cross-validation** on the training part:
 
```python
print(y_train.value_counts().sort_index())
 
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
```
 
Why only 2 folds? Look at the counts you just printed. The straight flush has only **2** hands in the training part. Every fold needs at least one hand of each class, so you can't use more folds than that.
 
---
 
## Step 2: Option 1, Class Weights
 
A class weight tells the model "a mistake on a rare class costs more". It changes nothing about the data.
 
```python
from sklearn.ensemble import RandomForestClassifier
 
rf_weighted = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
```
 
Logistic regression, decision trees and random forests all accept `class_weight="balanced"`.
 
**Optional: compute the weights yourself.** "Balanced" gives each class a weight of `total rows / (number of classes x rows in that class)`:
 
```python
counts = y_train.value_counts()
weights = len(y_train) / (len(counts) * counts)
print(weights.sort_index().round(2))
```
 
Which hand types get the biggest weights? Why?
 
---
 
## Step 3: Option 2, Oversampling
 
Oversampling repeats rare-class examples in the **training data** so the model sees them more often. Here's a short function using scikit-learn's `resample`:
 
```python
from sklearn.utils import resample
 
def oversample(X, y, seed=42):
    """Repeat rare-class rows until every class is as big as the largest one."""
    target = y.value_counts().max()
    parts_X, parts_y = [], []
    for cls in y.unique():
        X_c, y_c = X[y == cls], y[y == cls]
        if len(y_c) < target:
            X_c, y_c = resample(X_c, y_c, replace=True, n_samples=target, random_state=seed)
        parts_X.append(X_c)
        parts_y.append(y_c)
    return pd.concat(parts_X), pd.concat(parts_y)
```
 
Read it line by line and explain to yourself what each class ends up with. (`replace=True` means the same hand can be picked more than once.)
 
> A class with only 2 or 3 real hands becomes thousands of copies of those same hands. The model can end up memorizing them. Keep that in mind when you judge the results.
 
### The leakage rule (important)
 
> **Oversample only the training part, never before splitting.**
 
Why? If you copy the rare hands first and split afterward, copies of the same hand end up in both training and validation. The model "predicts" a hand it has already seen, and your score is fake.
 
The same trap exists **inside cross-validation**. If you oversample all the training data once and then cross-validate, copies leak between folds. So you oversample **inside each fold**, on that fold's training rows only:
 
```python
from sklearn.metrics import f1_score
 
fold_scores = []
for train_idx, val_idx in cv.split(X_train_fe, y_train):
    X_tr, y_tr = X_train_fe.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train_fe.iloc[val_idx], y_train.iloc[val_idx]
 
    X_tr_over, y_tr_over = oversample(X_tr, y_tr)     # training rows of this fold only
 
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_tr_over, y_tr_over)
    pred = model.predict(X_va)                        # validation rows stay untouched
    fold_scores.append(f1_score(y_va, pred, average="macro", zero_division=0))
 
print(np.mean(fold_scores), np.std(fold_scores))
```
 
(`.iloc` picks rows by position, which is what `cv.split` returns.)
 
**Questions:**
 
1. In your own words, why would oversampling before splitting inflate the score?
2. In the loop above, which rows get oversampled and which stay untouched? Why does that keep the score honest?
3. Optional experiment: oversample **all** of `X_train` (the raw columns) first, then score a random forest on it with `cross_val_score` (shown in Step 4). How much higher is it than the honest score? That gap is the leakage.
---
 
## Step 4: Compare Strategies
 
Pick one model (a random forest) and compare three ways of handling imbalance with cross-validated macro-F1:
 
| Strategy | How |
|---|---|
| Do nothing | A plain random forest |
| Class weights | `rf_weighted` from Step 2 |
| Random oversampling | Your `oversample` function, applied inside each fold |
 
The first two work directly with scikit-learn's cross-validation:
 
```python
from sklearn.model_selection import cross_val_score
 
for name, model in {
    "do nothing": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "class weights": rf_weighted,
}.items():
    scores = cross_val_score(model, X_train_fe, y_train, cv=cv, scoring="f1_macro")
    print(name, round(scores.mean(), 4), round(scores.std(), 4))
```
 
**Your turn:** put the loop from Step 3 inside a function `cv_oversample(model, X, y)` that returns the mean and standard deviation of macro-F1, so you can reuse it.
 
> **Hint**
> 1. Paste the loop inside the function and use `X` and `y` instead of `X_train_fe` and `y_train`.
> 2. Each fold needs a fresh, untrained model. Add `from sklearn.base import clone` and use `m = clone(model)` inside the loop, then fit `m`.
> 3. End with `return np.mean(fold_scores), np.std(fold_scores)`.
 
**Run all three strategies on both feature sets:** once on the **raw** columns (`X_train`) and once on your **engineered** features (`X_train_fe`). Put the six results in one table:
 
| Strategy | Raw features | Engineered features |
|---|---|---|
| Do nothing | | |
| Class weights | | |
| Random oversampling | | |
 
**Questions:**
 
1. Which strategy helps most? Is the gain bigger than the spread between folds?
2. Did the strategy help the rare classes? Did it hurt the common ones?
3. Compare the two columns of your table. Did the imbalance strategies matter more on the raw features or on the engineered features? What does that tell you about when imbalance handling is needed?
4. Compare how much the Week 03 features helped with how much the best imbalance strategy helped. Which mattered more?
---
 
## Step 5: Compare Models
 
Now compare several models on your engineered features:
 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
models = {
    "logistic regression": LogisticRegression(max_iter=2000),
    "decision tree": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient boosting": GradientBoostingClassifier(random_state=42),
    "neural network": make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)),
}
```
 
**Your turn:** for each model, report the mean and standard deviation of cross-validated macro-F1 with "do nothing" (`cross_val_score`) and with oversampling (`cv_oversample`), plus a note on training time. Then check the per-class recall on the validation set (`X_val_fe`, `y_val`) for your best model with `classification_report`, like in Week 02.
 
> Gradient boosting and `MLPClassifier` don't accept `class_weight`. Oversampling works with every model, because it changes the data and not the model.
 
### Optional challenge: can a model learn it without your features?
 
Train an `MLPClassifier` on the **raw** columns, but add extra training data by shuffling card positions within each hand (the shuffled hand has the same label). Does it get close to the engineered-feature models? What does this say about how much data a model needs to discover the structure on its own?
 
---
 
## Step 6: Lock Your Final Choice
 
Write down in a Markdown cell:
 
> **Final model:** (algorithm, settings), using engineered features, with (imbalance strategy). Chosen because ...
 
Fit it on the **full training part** (`X_train_fe`, `y_train`) and save it with `joblib.dump` to `results/`. Do not evaluate on the test file yet.
 
```python
import joblib
 
final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)   # use your own choice
final_model.fit(X_train_fe, y_train)
 
joblib.dump(final_model, "../results/final_model.joblib")
```
 
Week 05 loads this exact file name.
 
Two things to know about saved models. Only load a `.joblib` file you made yourself or trust, because loading a file can run code. And a model saved with one version of scikit-learn may not load in another, so keep your `requirements-lock.txt` from Week 06.
 
---
 
## Deliverable
 
`04_imbalance_models.ipynb`, containing:
 
- [ ] Stratified cross-validation with a fixed seed, and why you used 2 folds
- [ ] A comparison of three imbalance strategies on both raw and engineered features
- [ ] A written explanation of the leakage rule in your own words
- [ ] A comparison of at least four models, with mean and standard deviation of macro-F1
- [ ] A Markdown cell naming your final model and why
- [ ] The saved model in `results/`
- [ ] A short written answer (6-8 sentences): what helped, what didn't, what surprised you
---
 
## Common Mistakes
 
- **Oversampling before splitting.** This is the classic mistake. Oversample only the training rows, including inside each cross-validation fold.
- **Judging by one lucky validation score.** With so few rare examples, use cross-validation and report the spread.
- **Trusting the rare-class scores too much.** After oversampling, a class with 2 real hands is thousands of copies of them. Say how few real examples there are.
- **Comparing models on different splits.** Use the same `cv` for everything.
- **Looking for test file.** Ignore it until Week 05.
---
 
## Answers
 
- Training part: 3 four of a kind, 2 straight flushes, 3 royal flushes. That's why `n_splits=2`.
- Balanced weights: "Nothing" gets about 0.2. Four of a kind and royal flush get about 469, straight flush about 703.
- Imbalance strategies (random forest, macro-F1):
  - Raw: do nothing ≈ 0.12, class weights ≈ 0.14, oversampling ≈ 0.11
  - Engineered: do nothing ≈ 0.87, class weights ≈ 0.86, oversampling ≈ 0.87
- No strategy rescues the raw columns, and on engineered features they change almost nothing. The features (0.12 → 0.87) matter much more than any imbalance trick.
- Leakage: oversampling the raw training rows **before** cross-validation gives about 0.94, against an honest 0.11.
- Models (engineered, do nothing / oversampling):
  - Random forest ≈ 0.87 / 0.87
  - Neural network ≈ 0.87 / 0.86
  - Decision tree ≈ 0.86 / 0.86
  - Gradient boosting ≈ 0.86 / 0.86
  - Logistic regression ≈ 0.69 / 0.86
- The models are basically tied (std ≈ 0.01). Logistic regression is the only one that oversampling really helps.
- Every model stops around 0.86-0.87 for the same reason as Week 03: a straight flush and a royal flush look the same with the 7 main features. With the optional `is_royal` and `is_straight_flush`, a random forest reaches 1.0.
---
 
## Next Week
 
In Week 05, you'll test your final model once on the test file, study its mistakes, and think about what a perfect score would mean.
 
