# Week 03: Feature Engineering

**Project:** Classifying Poker Hands
**Notebook:** `notebooks/03_feature_engineering.ipynb`
**Code file:** `notebooks/features.py`

This is the heart of the project. Last week's models saw ten raw numbers and struggled. 
This week you'll describe each hand in a way a model can actually use like "how many cards share a rank" or "are all the suits the same" and see how much that changes things.

You write every feature yourself. Under each one there is a **"Try it yourself first"** block. Try the feature on your own first, and use the block only if you need a nudge. It describes the steps in plain words, not code.

**By the end of this week you will be able to:**

- Explain why the way data is represented can matter more than the choice of model
- Build features from raw columns using pandas and NumPy
- Move code from a notebook into a reusable function in `features.py`
- Measure whether new features actually help

Plan for ~ 2 to 3 hours for this week's content. This week is longer than others.

---

## Step 1: Why the Raw Columns Hide the Hands

Look back at your Week 02 notes. A hand is "a pair" when two ranks match, no matter **which** two cards they are. But the raw columns tie each rank to a fixed position (`C1`, `C2`, ...). The same hand can appear in 120 different orders, and the model has to learn each ordering separately.

Good features should:

- **Not depend on card order** (the same five cards always give the same features)
- **Make the key facts easy to see** (matches, runs, same suit)

Think first, then keep reading: if you could give the model only five or six numbers describing a hand, which would you choose? Write your ideas in a Markdown cell **before** looking at the features below.

---

## Step 2: Set Up Your Building Blocks

Open the notebooks folder, click **New → Python 3 (ipykernel)**, and name the notebook `03_feature_engineering`. 

Start with the same data and split as Week 02:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cols = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "hand"]
data = pd.read_csv("../data/raw/poker_train.csv", header=None, names=cols)

X = data.drop(columns="hand")
y = data["hand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rank_cols = ["C1", "C2", "C3", "C4", "C5"]
suit_cols = ["S1", "S2", "S3", "S4", "S5"]
```
Using same `random_state = 42` and `stratify = y` gives you exact same split as last week so compare fairly.

Two building blocks make most of the features easy. Work out how to make each one. (Week 01 warm-up has NumPy moves you need)

**Building block A: sorted ranks.** Sorting the five ranks of each hand, low to high, removes the effect of card order.

> **Try it yourself first**
>
> 1. Take the five rank columns as a NumPy array with one row per hand.
> 2. Sort each row from low to high (NumPy can sort along a row instead of along a column).
> 3. Check one hand by hand: do the sorted numbers match the five ranks you expect?

**Building block B: rank counts.** For each hand, count how many cards have each rank from 1 to 13. A pair shows up as a count of 2, a full house as a 3 and a 2.

> **Try it yourself first**
>
> 1. Make a table of zeros with one row per hand and 13 columns (one per rank).
> 2. For each rank from 1 to 13, count how many of the five rank columns in each row equal that rank.
> 3. Put each count in the matching column.
> 4. Check: every row should add up to 5, and a hand with a pair should have a 2 somewhere.

---

## Step 3: Build the Features

Write each one below. Test every feature on a few hands whose labels you know before moving on.

### Rank features

**`max_count`**: the biggest rank count in the hand (2 means a pair, 3 means three of a kind, 4 means four of a kind).

> **Try it yourself first**
>
> 1. Start from your rank counts.
> 2. For each hand, take the largest of the 13 counts.

**`n_distinct_ranks`**: how many different ranks appear (5 for no pair, 4 for one pair, and so on).

> **Try it yourself first**
>
> 1. Start from your rank counts.
> 2. For each hand, count how many of the 13 counts are bigger than zero.

**`n_pairs`**: how many ranks appear exactly twice.

> **Try it yourself first**
>
> 1. Start from your rank counts.
> 2. For each hand, count how many of the 13 counts are exactly 2.
> 3. Check: two pairs should give 2, and a full house should give 1.

**`has_three`**: is there a rank that appears exactly three times? (1 for yes, 0 for no)

> **Try it yourself first**
>
> 1. Start from your rank counts.
> 2. For each hand, check whether any of the 13 counts is exactly 3.
> 3. Turn the yes/no answer into 1 or 0.

### Suit features

**`n_distinct_suits`**: how many different suits appear in the hand.

> **Try it yourself first**
>
> 1. Take the five suit columns.
> 2. For each hand, count how many different suit values appear. (Compare each suit from 1 to 4 against the five columns, as you did for ranks.)

**`is_flush`**: 1 if all five cards share a suit, else 0.

> **Try it yourself first**
>
> 1. Use the number of distinct suits.
> 2. A hand is a flush exactly when that number is 1.
> 3. Turn the yes/no answer into 1 or 0.

### The straight feature (the hard one)

A straight is five ranks in a row. With sorted ranks, think about the **gaps** between neighbors, which NumPy can compute for you (look up `np.diff`).

Work these out before you write code:

1. What do the gaps look like for 4, 5, 6, 7, 8?
2. What's special about a straight compared with other hands? (Think about `n_distinct_ranks` and the spread from lowest to highest.)
3. **The Ace problem:** 10, J, Q, K, A has ranks 1, 10, 11, 12, 13 after sorting. The gaps are not all 1. How can you detect this case on its own?

**`is_straight`**: 1 if the five ranks form a straight, else 0.

> **Try it yourself first**
>
> 1. Normal straights: the hand has five different ranks, and the highest rank minus the lowest rank is exactly 4. Both conditions together are enough.
> 2. The Ace-high straight is the special case. After sorting, the ranks are exactly 1, 10, 11, 12, 13. Check for that set of five numbers directly.
> 3. A hand is a straight if either the normal check or the Ace-high check is true.
> 4. Test it: print rows where it's 1 and confirm they're class 4, 8, or 9. Then look for class 4, 8, or 9 rows where it's 0. There should be none.

### Optional extras

Add any of these if you want them:

- The highest rank in the hand
- Counts of each suit
- `is_straight_flush` and `is_royal`

> **Try it yourself first**
>
> - `is_straight_flush`: combine the straight feature and the flush feature.
> - `is_royal`: a straight flush that is also the Ace-high straight.

---

## Step 4: Move It into `features.py`

You'll use these features in every notebook from now on, so put them in one function instead of copying code around. In the `notebooks` folder, create a plain text file named `features.py` (in Jupyter: **New → Text File**, then rename it to `features.py`). It should start like this:

```python
# features.py
import numpy as np
import pandas as pd

def make_features(df):
    """Return a table of order-independent features for poker hands."""
    # your code here
    return features
```

The function takes a DataFrame with the ten raw columns and returns a DataFrame with your new features, one column per feature. Import it in the notebook:

```python
from features import make_features

X_train_fe = make_features(X_train)
X_val_fe = make_features(X_val)
X_train_fe.head()
```

> After you edit `features.py`, restart the kernel (**Kernel → Restart Kernel**) before importing it again. Notebooks keep the old version until you do. The notebook and `features.py` must be in the same folder.

Check your work on a few hands where you know the answer, like a made-up flush, before running on the whole dataset.
Then test that card order doesn’t matter: swap two cards in one hand (both the suit and the rank) and check that the features stay exactly the same.

---

## Step 5: Retrain and Compare

Use the same split and the same models as Week 02, now on `X_train_fe` and `X_val_fe`. Rebuild your results table with accuracy, macro-F1, and per-class recall, now with **two** rows for each model: raw features and engineered features.

Save the confusion matrix of your best engineered model to `reports/figures/` 
> **Hint** use `plt.savefig(...)` before `plt.show()`

---

## Step 6: What Changed?

Answer in writing:

1. How much did macro-F1 improve for each model? Which model gained the most?
2. Which classes improved? Did any classes **not** improve? What does that tell you about the features you built?
3. Did logistic regression catch up to the tree-based models? Why might that be?
4. If a hand is deterministic, why can't the model reach 100% even now? (Check your confusion matrix for the specific mistakes.)
5. What does this say about the importance of features compared with the choice of model?

---

## Deliverable

`03_feature_engineering.ipynb` and `features.py`, containing:

- [ ] Your own written ideas for features, written before building them
- [ ] Both building blocks (sorted ranks and rank counts), checked on a few hands
- [ ] A `make_features` function in `features.py` with at least 6 features
- [ ] Checks showing the features are correct on known hands, and that card order doesn't change them
- [ ] A raw-versus-engineered comparison table for all 3 models
- [ ] A confusion matrix for your best engineered model, saved to `reports/figures/`
- [ ] A short written answer (8-10 sentences) to the Step 6 questions

---

## Common Mistakes

- **Features that depend on card order.** Test it: shuffle the card positions of one hand and check the features don't change.
- **Forgetting the Ace-high straight.** Your `is_straight` will miss royal flushes and 10-J-Q-K-A straights unless you handle it.
- **Treating suit numbers as amounts.** Suit 4 isn't more than suit 1. Count and compare suits, don't add them.
- **Copy-pasting the feature code.** Use the function in `features.py`, so training and later test data are processed identically.

---

## Check your answers

Your exact numbers depend on which features you built, so these are for the 7 main features (max_count, n_distinct_ranks, n_pairs, has_three, n_distinct_suits, is_flush, is_straight).

- `is_straight` is 1 for every class 4, 8 and 9 hand, and 0 for every other hand.
- With the 7 features, accuracy jumps from about 0.47-0.58 to about **1.00** for all three models. Each model gets just **1 hand wrong** out of 3,515.
- That one hand is the only straight flush in validation (2, 3, 4, 5, 6, all hearts). The tree and the forest call it a royal flush, and logistic regression calls it a flush.
- Why? With these 7 features, a straight flush and a royal flush look exactly the same: both are a straight and a flush. Nothing tells the model how high the cards are. On top of that, training only has 2 straight flushes to learn from.
- Macro-F1 is about 0.88 for logistic regression and about 0.80 for the tree and the forest. It looks low for a model with 1 mistake, because every class counts equally. Missing the only straight flush makes that class's F1 equal 0, and the tree and forest also get an F1 of 0 for "royal flush", since their one royal flush guess was wrong.
- If you add `is_royal` (and `is_straight_flush`), the tree and the forest get every validation hand right. Logistic regression still misses the straight flush.
- Logistic regression goes from the worst model (0.07) to the best with the 7 features. Same model, better description of the data.

---

## Next Week

In Week 04, you'll deal with the rare hands directly (class weights and oversampling), learn the leakage mistake that makes scores look better than they really are, and compare more models.
