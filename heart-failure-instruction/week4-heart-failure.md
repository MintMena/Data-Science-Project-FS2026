# Week 04: Compare Many Models and Find Key Features
 
**Project:** Predicting Heart Failure Survival
**Notebook:** `notebooks/04_compare_models.ipynb`
 
This week you try about eleven classifiers, compare them fairly, and find out which measurements matter most.
 
**By the end of this week you will be able to:**
 
- Use a dictionary and a loop to train many classifiers
- Read the variation in cross-validation scores
- Rank features with more than one method
- 
Plan for about ~ 2 hours.
 
---
 
## Step 1: Set Up
 
- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `04_compare_models`
Load the prepared table from Week 03 (the log steps are already done in it), and copy your `rcv` and `scoring` objects:
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
 
data = pd.read_csv("../data/processed/patients.csv")
FEATURES = [c for c in data.columns if c not in ("time", "DEATH_EVENT")]
X_train = data[FEATURES]
y_train = data["DEATH_EVENT"]
 
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {"MCC": make_scorer(matthews_corrcoef), "AUC": "roc_auc", "Recall": "recall"}
```
 
---
 
## Step 2: Your Turn - Build a Dictionary of Classifiers
 
Store your models in a dictionary, with the model name as the key and the model as the value, for example `{"Logistic Regression": model1, "Random Forest": model2}`. Include these models:
 
| Model | Needs scaling? | How to balance the classes |
|---|---|---|
| Logistic Regression | Yes | `class_weight="balanced"` |
| K-Nearest Neighbours | Yes | not available |
| Support Vector Machine (`SVC`) | Yes | `class_weight="balanced"`, plus `probability=True` |
| Naive Bayes (`GaussianNB`) | Yes | not available |
| Decision Tree | No | `class_weight="balanced"` |
| Random Forest | No | `class_weight="balanced"` |
| Extra Trees | No | `class_weight="balanced"` |
| Gradient Boosting | No | not available |
| XGBoost | No | `scale_pos_weight=2.1` (survivors ÷ deaths) |
| LightGBM | No | `class_weight="balanced"` |
| Neural network (`MLPClassifier`) | Yes | not available |
 
Wrap the models that need scaling in `make_pipeline(StandardScaler(), model)`, and give every model that accepts it `random_state=42`.
 
Imports you'll need:
 
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
```
 
> **Hint**
> 1. `probability=True` lets the SVM give probabilities, which ROC-AUC needs.
> 2. Use `MLPClassifier(max_iter=2000, random_state=42)` so it has time to finish learning.
> 3. Use `LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, n_jobs=1)` so it doesn't print lots of messages or run very slowly.
 
---
 
## Step 3: Your Turn - Write the Loop
 
For each model, run `cross_validate` with your `rcv` and `scoring`, and save the average MCC, ROC-AUC and recall, plus the std of MCC. Turn the results into a table sorted by MCC.
 
> **Hint**
> 1. Make an empty list `rows = []` and an empty dictionary `all_mcc = {}` before the loop
> 2. Inside the loop: `scores = cross_validate(model, X_train, y_train, cv=rcv, scoring=scoring, n_jobs=-1)`
> 3. Save all 50 MCC scores with `all_mcc[name] = scores["test_MCC"]` (you need them for the box plot)
> 4. Append a dictionary with the name and the averages to `rows`, then `results = pd.DataFrame(rows).sort_values("MCC", ascending=False)`
 
Then draw a box plot of each model's 50 MCC scores. This shows how much the score jumps between folds:
 
```python
order = results["model"]
plt.figure(figsize=(8, 6))
plt.boxplot([all_mcc[name] for name in order], tick_labels=order, vert=False)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("MCC (50 cross-validation scores)")
plt.tight_layout()
plt.savefig("../reports/figures/model_mcc_boxplot.png")
plt.show()
```
 
If you get an error about `tick_labels`, change it to `labels`.
 
---
 
## Step 4: Rank the Features
 
Use at least two of these methods and see whether they agree:
 
- **Mutual information:** how much each feature tells you about the outcome (0 = nothing).
```python
from sklearn.feature_selection import mutual_info_classif
 
mi = mutual_info_classif(X_train, y_train, random_state=42)
pd.Series(mi, index=FEATURES).sort_values(ascending=False)
```
 
- **Random Forest importance:** after fitting a forest, read `.feature_importances_`.
```python
forest = RandomForestClassifier(class_weight="balanced", random_state=42).fit(X_train, y_train)
pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
```
 
- **Permutation importance:** shuffle one feature at a time and measure how much worse the model gets. Use logistic regression here, because a forest memorizes its training patients.
```python
from sklearn.inspection import permutation_importance
 
lr = models["Logistic Regression"].fit(X_train, y_train)
perm = permutation_importance(lr, X_train, y_train, n_repeats=30, random_state=42, scoring="roc_auc")
pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
```
 
**Questions to answer in writing:**
 
1. Which models did best? Did complex models (boosting, the neural network) beat simple ones?
2. How much do scores vary between folds? What does that say about small datasets?
3. Which model has the highest recall? Is it the same as the one with the highest MCC?
4. Which features come out on top? Does that match what you found in Week 01?
---
 
## Deliverable
 
`04_compare_models.ipynb`, containing:
 
- [ ] At least 8 classifiers compared in one table (MCC, MCC std, ROC-AUC, recall)
- [ ] A box plot of MCC scores for each model, saved to `reports/figures/`
- [ ] Features ranked with at least two methods
- [ ] Written answers to the questions
---
 
## Common Mistakes
 
- **Treating small score differences as real.** Look at the box plot before you declare a winner.
- **Forgetting to balance the classes.** Without it, some models predict "survived" for almost everyone.
- **Including `time`.** `FEATURES` should have 11 columns.
- **Forgetting `probability=True` on the SVM.** Without it, ROC-AUC can't be computed.
---
 
## Answers
 
- MCC (highest first):
  - Random Forest ≈ 0.42 (ROC-AUC ≈ 0.79, recall ≈ 0.61)
  - Naive Bayes ≈ 0.41 (recall only ≈ 0.47)
  - SVM ≈ 0.39
  - XGBoost, LightGBM, Logistic Regression ≈ 0.35-0.36
  - Gradient Boosting ≈ 0.33
  - Decision Tree, Extra Trees ≈ 0.31
  - KNN ≈ 0.20, Neural network ≈ 0.19
- MCC varies a lot between folds (std ≈ 0.13). The top 4-5 models are a tie.
- Boosting and the neural network don't help. With 209 patients, there isn't enough data for them.
- Logistic regression has the highest recall (≈ 0.65). A doctor would usually care most about not missing deaths.
- Top features: `serum_creatinine` is first in all three methods, then `age` and `ejection_fraction`.
- The methods disagree on the rest: forest importance ranks `platelets` and `creatinine_phosphokinase` fairly high, but mutual information gives them 0.
---

![heart-dis](images/heart-dis.webp)
 
## Next Week
 
In Week 05, you'll test the research paper's claim and tune your best models.
 
