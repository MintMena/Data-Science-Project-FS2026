# Week 05: Evaluate and Analyze Errors

**Project:** Classifying Poker Hands
**Notebook:** `notebooks/05_evaluation_errors.ipynb`

This is the week you open the test file. You'll score your final model once, study its mistakes, and measure how far it is from a perfect score.

**By the end of this week you will be able to:**

- Evaluate a final model on held-out data with the right metrics
- Read a confusion matrix and explain the specific mistakes
- Compare a model's score with a perfect score and explain the gap
- Say honestly what the model learned and what it didn't

Plan for about 1.5 hours.

---

## Step 1: The One Test Evaluation

1. Download `poker_test.csv` from the club GitHub (same place as Week 01)
   https://github.com/MintMena/Data-Science-Project-FS2026
2. Move the file into `data/raw/`

- Open the notebooks folder
- Click **New → Python 3 (ipykernel)**
- Name the notebook `05_evaluation_errors`
- Make sure `features.py` is in the same folder

Load the test file with the same column names, and your saved model:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score, ConfusionMatrixDisplay
from features import make_features

cols = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "hand"]
test = pd.read_csv("../data/raw/poker_test.csv", header=None, names=cols)

X_test = test.drop(columns="hand")
y_test = test["hand"]

model = joblib.load("../results/final_model.joblib")
pred = model.predict(make_features(X_test))
```

If your final model uses engineered features, apply `make_features` exactly as you did for training. This is why the function lives in `features.py`.

Then compute accuracy, macro-F1, and a full report:

```python
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro-F1:", round(f1_score(y_test, pred, average="macro", zero_division=0), 4))
print(classification_report(y_test, pred, zero_division=0))
```

Do this **once**. If you change your model after seeing these numbers, the test set no longer gives an honest score.

**Questions:**

1. How does the test macro-F1 compare with your cross-validated macro-F1 from Week 04? A big drop would be a warning sign.
2. Count how many hands of each class the test file has (`value_counts`). It has only 1 hand each of classes 7, 8 and 9. What does that mean for how much you can trust their recall?
3. What's your accuracy next to the dummy model's? Which number tells you more?

---

## Step 2: The Confusion Matrix

```python
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, normalize="true", values_format=".2f"
)
plt.savefig("../reports/figures/test_confusion_matrix.png")
plt.show()
```

Each row is the **real** class, each column is what the model **predicted**. Numbers off the diagonal are mistakes.

**Questions:**

1. Which pairs of classes get confused? Does it make sense?
2. Which classes are almost always right? Which are the weakest?
3. Are the mistakes on rare classes caused by too few training examples, or by features that don't capture the difference?

---

## Step 3: Study the Mistakes

Pull out the wrong predictions and read the actual hands:

```python
errors = test.copy()
errors["predicted"] = pred
wrong = errors[errors["hand"] != errors["predicted"]]

wrong
```

For each mistake, write the hand out as real cards and decide: **why** might the model have missed it? Was it a case your features didn't cover? Fixing a feature now would be going back to tune on test data, so don't do that. Instead, **write down what you'd change** for a future version.

---

## Step 4: Compare With a Perfect Score

In this dataset, every label comes from exact rules. Imagine a program that simply followed the poker rules you wrote in Week 01. It would check each hand against the rules, strongest hand first, and give it the right label every time. It would score **100% accuracy and a macro-F1 of 1.0**, with no training data at all.

You don't need to write that program. Use it as a yardstick.

**Questions:**

1. How far is your model from a perfect score on accuracy? On macro-F1? Which classes make up most of the gap?
2. Look back at the mistakes from Step 3. Would a program that follows the rules make them? Why or why not?
3. Which of your features already do part of the rule-following for the model? What did the model still have to learn by itself?
4. **Think about this:** if the rules for poker hands are already known, why would anyone use machine learning here? What did this project teach you that a rule-following program wouldn't?

---

## Step 5: Learning Curve

Check how much training data your model needs. `learning_curve` trains your model on growing slices of the **training** data and scores it with cross-validation. It never touches the test file.

Rebuild the training part from Week 04 first:

```python
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../data/raw/poker_train.csv", header=None, names=cols)
X = data.drop(columns="hand")
y = data["hand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_fe = make_features(X_train)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_train_fe, y_train,
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    cv=cv, scoring="f1_macro", shuffle=True, random_state=42, n_jobs=-1,
)

plt.plot(sizes, val_scores.mean(axis=1), marker="o")
plt.xlabel("training hands")
plt.ylabel("validation macro-F1")
plt.savefig("../reports/figures/learning_curve.png")
plt.show()
```

Use your own final model in place of the random forest if it's different. The sizes are fractions of each fold's training rows.

Is the curve still rising, or flat? Would more data help? Run it once more with `X_train` (raw columns) instead of `X_train_fe` and compare.

---

## Step 6: What Can't This Model Tell You?

Write an honest list of limits:

- The data has no strategy or context. It doesn't tell you how to play a hand.
- The training file has only a few examples of the rarest hands. How confident can you be in your model on them?
- How would the model do if suits or ranks were coded differently?
- Is a machine learning model the right tool here? Why or why not?

---

## Step 7: Check In

When you finish, check in so we know you're still with us. It takes 1 minute (you need a free GitHub account):

1. Open the [check-in form](https://github.com/MintMena/Data-Science-Project-FS2026/issues/new?template=checkin.yml)
2. Pick **Poker Hand** and **Week 05**, and type your name (a first name is fine, it shows publicly)
3. Say whether you'll join the model contest (the date is coming soon), then click **Create**

A bot replies to confirm.

---

## Deliverable

`05_evaluation_errors.ipynb`, containing:

- [ ] A single test evaluation, compared with cross-validation and the dummy baseline
- [ ] A confusion matrix with a written reading of it
- [ ] A look at specific mistakes, with an explanation for each
- [ ] A written comparison of your model's scores with a perfect score, with your answers to the Step 4 questions
- [ ] A learning curve with a one-paragraph interpretation
- [ ] A written "limits of this model" section (6-10 sentences)
- [ ] Your Week 05 check-in

---

## Common Mistakes

- **Evaluating on the test file more than once.** After that, you're tuning to it.
- **Fixing features after seeing test errors.** Note ideas for next time instead.
- **Reporting accuracy as the headline.** Lead with macro-F1 and per-class recall.
- **Skipping the gap.** Don't stop at "my score is high". Say exactly what is missing from a perfect score, and why.
- **Reading the learning curve on the test set.** Use validation data.

---

## Answers

- Test file: 4,958 hands. Only 1 four of a kind, 1 straight flush and 1 royal flush.
- Random forest with the 7 main features: accuracy ≈ 0.9998, macro-F1 ≈ 0.87. That matches the cross-validated ≈ 0.87 from Week 04, so no warning sign.
- Dummy accuracy ≈ 0.50.
- There is exactly **1 mistake**: the straight flush (5, 6, 7, 8, 9, all the same suit) is called a royal flush. Every other class has recall 1.0.
- Because of that one hand, the straight flush gets F1 = 0 and the royal flush gets F1 ≈ 0.67 (1 right, 1 wrong guess). That's why macro-F1 is 0.87 while accuracy is almost 1.0.
- A rule-following program would never make this mistake, because it checks whether the high card is an Ace. The 7 features never tell the model how high the cards are.
- Learning curve (engineered): ≈ 0.53, 0.73, 0.83, 0.83, 0.86. Still rising a little, because more data means more rare hands to learn from.
- Learning curve (raw): ≈ 0.10-0.12 at every size. More data barely helps without good features.

---

## Next Week

In Week 06, you'll write up the project, package it so someone else can reproduce your results, and get ready for the model contest.
