# Week 05: Test the Paper's Claim and Tune

**Project:** Predicting Heart Failure Survival
**Notebook:** `notebooks/05_paper_and_tuning.ipynb`

This week you run your own experiment to check a published research claim, show why leakage makes scores look better than they really are, and tune your best models.

**By the end of this week you will be able to:**

- Test a research claim with your own experiment
- Demonstrate data leakage
- Tune a classifier for MCC

Plan for about 2 hours.

---

## The Claim

The research paper behind this dataset (Chicco and Jurman, 2020) claims that serum creatinine and ejection fraction alone are enough to predict survival. Both are cheap, routine blood and heart tests.

---

## Step 1: Two Features vs All Features

- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `05_paper_and_tuning`

Copy your setup cell from Week 04 (loading `patients.csv`, `FEATURES`, `rcv` and `scoring`) to the top. Then re-run Logistic Regression, Random Forest and SVM (same settings as Week 04) with two feature sets:

1. All 11 features
2. Only `serum_creatinine` and `ejection_fraction`

To use only two features, pass `X_train[["serum_creatinine", "ejection_fraction"]]` instead of `X_train`. Put both sets of results (MCC, ROC-AUC, recall) in one table.

---

## Step 2: The Leakage Demonstration

In a separate table clearly labelled **"for reference only"**, add `time` to the features and score Random Forest again:

```python
from sklearn.ensemble import RandomForestClassifier

X_leaky = data[FEATURES + ["time"]]
leaky = cross_validate(RandomForestClassifier(class_weight="balanced", random_state=42),
                       X_leaky, y_train, cv=rcv, scoring=scoring, n_jobs=-1)
print("MCC:", leaky["test_MCC"].mean(), "AUC:", leaky["test_AUC"].mean())
```

Explain in writing why the jump in score is fake.

---

## Step 3: Tune

Use `RandomizedSearchCV` with MCC as the score on Random Forest:

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 3, 5, 8],
    "min_samples_leaf": [1, 3, 5, 8, 12],
    "max_features": ["sqrt", 0.5, 0.8],
}

forest = RandomForestClassifier(class_weight="balanced", random_state=42)
search = RandomizedSearchCV(forest, grid, n_iter=40,
                            scoring=make_scorer(matthews_corrcoef),
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

This takes about a minute.

**Your turn:** tune at least one more model, for example the `C` setting of Logistic Regression or SVM (try 0.001, 0.01, 0.1, 1, 10, 100). Then re-score every tuned model with the same repeated CV (`rcv`) from Week 03, so the comparison with Week 04 is fair.

> **Hint**
> 1. Only one setting to try? Use `GridSearchCV` instead of `RandomizedSearchCV`. It tries every value.
> 2. The model sits inside a pipeline, so the setting name is `logisticregression__C` or `svc__C`.
> 3. To re-score: `cross_validate(search.best_estimator_, X_train, y_train, cv=rcv, scoring=scoring)`

**Questions to answer in writing:**

1. Does the paper's claim hold? What would that mean for doctors?
2. How much did `time` "improve" the score, and why can't you use it?
3. The search's best MCC and the repeated-CV MCC of the same model are different. Why? Which one do you trust?
4. Which model will you choose as your final model, and why?

---

## Step 4: Check In

When you finish, check in so we know you're still with us. It takes 1 minute (you need a free GitHub account):

1. Open the [check-in form](https://github.com/MintMena/Data-Science-Project-FS2026/issues/new?template=checkin.yml)
2. Pick **Heart Failure** and **Week 05**, and type your name (a first name is fine, it shows publicly)
3. Say whether you'll join the model contest (the date is coming soon), then click **Create**

A bot replies to confirm.

---

## Deliverable

`05_paper_and_tuning.ipynb`, containing:

- [ ] A table comparing two features with all 11
- [ ] A labelled leakage demonstration with a written explanation
- [ ] At least two models tuned and re-scored with repeated CV
- [ ] A final model chosen by CV, with a written reason
- [ ] Your Week 05 check-in

---

## Common Mistakes

- **Keeping `time` in your final model** because its score looks good.
- **Comparing tuned models with a different CV setup** than the untuned ones.
- **Trusting the search's best score.** It picked the luckiest settings on those exact folds. Re-score with `rcv`.

---

## Answers

- Two features vs all 11 (MCC):
  - Logistic Regression ≈ 0.35 vs ≈ 0.35 (same)
  - SVM ≈ 0.44 vs ≈ 0.39 (two features are **better**)
  - Random Forest ≈ 0.35 vs ≈ 0.42 (two features are worse)
- The claim mostly holds: two cheap tests do about as well as all 11 for the simpler models. Random Forest gets a bit more out of the extra columns.
- With `time`: MCC ≈ 0.61, ROC-AUC ≈ 0.89. That jump is leakage. A doctor doesn't know how long a new patient will be followed.
- Tuned Random Forest: `n_estimators=800`, `max_depth=5`, `min_samples_leaf=3`, `max_features=0.8`.
  - Search best MCC ≈ 0.51, but with repeated CV ≈ 0.47 (ROC-AUC ≈ 0.79, recall ≈ 0.68). The search's score is too optimistic because it picked the luckiest settings on those 5 folds.
- Tuning `C` barely changes Logistic Regression (best `C=10`, MCC ≈ 0.36) or SVM (best `C=1`, MCC ≈ 0.39).
- The tuned Random Forest has the best repeated-CV score (≈ 0.47 vs ≈ 0.42 untuned), so it's a good final model.

---

## Next Week

In Week 06, you'll test your final model, write your conclusions, and get ready for the model contest.
