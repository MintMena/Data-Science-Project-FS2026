# Week 01: Explore the Data

**Project:** Classifying Poker Hands
**Notebook:** `notebooks/01_eda.ipynb`

This week we'll take a look at the dataset!

Before we build any model, we want to understand 2 things:
1. What a poker hand actually is
2. What our dataset looks like

You will find one big problem with this data and it will follow us for the rest of the project so take your time here.

**By the end of this week you will be able to:**

- Name the ten poker hands and explain what makes each one
- Load a dataset that has no header row and name its columns
- Count how many hands of each type we have and explain why that is a problem
- Explain why the raw columns make the hands hard to see

Expected time for content ~ 1 hour

---

## A Quick Poker Primer

You don't need to have played poker. Everything you need is here.

**The cards.** A deck has 52 cards. Each card has a **suit** (hearts, diamonds, clubs, or spades) and a **rank**: Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, or King. There are 4 suits and 13 ranks.

**A hand** is five cards. The order the cards are dealt in doesn't matter. Only which five cards you hold does.

**The hands.** Poker has ten kinds of hand. Here they are from weakest to strongest, with the class number this dataset uses for each:

| Class | Hand | What it means | Example |
|---|---|---|---|
| 0 | Nothing | None of the hands below | 2, 5, 8, Jack, King, mixed suits |
| 1 | One pair | Two cards of the same rank | 7, 7, 3, 9, King |
| 2 | Two pairs | Two different pairs | 7, 7, 3, 3, King |
| 3 | Three of a kind | Three cards of the same rank | 7, 7, 7, 3, King |
| 4 | Straight | Five ranks in a row, any suits | 4, 5, 6, 7, 8 |
| 5 | Flush | Five cards of the same suit, any ranks | all hearts: 2, 5, 9, Jack, King |
| 6 | Full house | Three of a kind plus a pair | 7, 7, 7, 3, 3 |
| 7 | Four of a kind | Four cards of the same rank | 7, 7, 7, 7, King |
| 8 | Straight flush | A straight, all in one suit | all spades: 4, 5, 6, 7, 8 |
| 9 | Royal flush | 10, Jack, Queen, King, Ace, all in one suit | all clubs: 10, J, Q, K, A |

**Two things to know**

- **A hand gets its best label.** A full house also contains a pair and three of a kind, but it counts as a full house. A straight flush is also a straight and a flush, and counts as a straight flush.
- **The Ace can be low or high in a straight.** Ace, 2, 3, 4, 5 is a straight, and so is 10, Jack, Queen, King, Ace. A straight can't wrap around, though. Queen, King, Ace, 2, 3 is not a straight.

---

## The Problem

Given five playing cards, can a model tell what poker hand it is?

In this dataset, each card has a **suit** coded 1 to 4 and a **rank** coded 1 to 13, where 1 is Ace, 11 is Jack, 12 is Queen, and 13 is King. A hand has 10 input columns, and the target is the **hand type**, the class number from 0 to 9 in the table above.

Because the target is a category, this is a **classification** problem. Unlike most real data, the label here follows exact rules. So the interesting question isn't "is there a pattern?" but **"how much of that rule can a model find on its own, and how much do we have to help it?"**

**How the dataset writes a card**
Each card is 2 numbers :
- **Suit** from 1 to 4
- **Rank** from 1 to 13 where 1 = Ace, 11 = Jack, 12 = Queen, 13 = King
(Follow the order)

---

## Step 1: Get the Data
You don't need to search for anything. We already prepared the file for you.

1. Download the dataset poker_train.csv (github link -> poker/poker_train.csv)
   https://github.com/MintMena/Data-Science-Project-Dataset-FS2026
   > On Github, click the file name then click download button ("Download raw file") on top        right of the file view
2. Move the file into `data/raw/` folder inside your poker-hand-project folder from Week 00
> **Never edit files in `data/raw/**

> To check that it's in there.

   > Windows : terminal -> dir data\raw

   > Mac/Linux : terminal -> ls data/raw

The exact file names may differ a little. Use the names you actually see. Open each file in a text editor if you like: if the first line is already a hand, there is no header row.

---

## Warm-Up: The Python You'll Need (optional)

The guides give you code to run, and later weeks ask you to write some yourself. This warm-up covers the Python you'll use. If you already know it, skip ahead to Step 2.

Start Jupyter (see Week 00) and make a practice notebook in `notebooks/` called `01_warmup`. It's just for practice and isn't part of your deliverable. Type each example into a cell, run it with `Shift + Enter` (or the **Run** button), and change it to see what happens.

### Variables and printing

A variable is a name for a value.

```python
temperature = 25.5
unit = "C"
print(temperature, unit)
print(temperature * 2)
```

### Lists

A list holds several values in order. Counting starts at **0**, not 1.

```python
readings = [20.1, 25.5, 30.2, 18.7]
print(len(readings))      # how many items
print(readings[0])        # the first item
print(readings[-1])       # the last item
print(readings[1:3])      # items 1 and 2 (the end is not included)
```

### Loops

A loop repeats code for each item. The indented lines are the part that repeats.

```python
for r in readings:
    print(r, "is above 22" if r > 22 else "is 22 or below")

for k in range(1, 4):     # 1, 2, 3 (the end is not included)
    print(k)
```

### Functions

A function packages code so you can reuse it. `return` sends back a result.

```python
def to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32

print(to_fahrenheit(25.5))
```

### Dictionaries

A dictionary stores values under names, called keys.

```python
scores = {"model A": 0.81, "model B": 0.87}
print(scores["model A"])

for name, score in scores.items():
    print(name, score)
```

You'll use this later to keep several models together and loop over them.

### A table in pandas

pandas holds a table called a **DataFrame**. Each column is picked by name, and you can filter rows with a condition.

```python
import pandas as pd

df = pd.DataFrame({"temp": [20.1, 25.5, 30.2, 18.7], "output": [460, 445, 430, 468]})

print(df["temp"])                   # one column
print(df["temp"].mean())            # an average
print(df[df["temp"] > 22])          # only the rows where temp is above 22
print(df[df["temp"] > 22].shape)    # (rows, columns) of that result
```

### Arrays in NumPy

NumPy holds grid-shaped numbers called **arrays**. Poker hands are a good fit, since each row is one hand. You'll use these moves in Week 03.

```python
import numpy as np

hands = np.array([[7, 7, 3, 9, 13],
                  [4, 5, 6, 7, 8]])

print(hands.shape)                 # (2 rows, 5 columns)
print(np.sort(hands, axis=1))      # sort within each row
print(hands == 7)                  # True where a value equals 7
print((hands == 7).sum(axis=1))    # how many 7s in each row
print(hands[:, 0])                 # the first column
```

`axis=1` means "work across each row". `axis=0` means "work down each column". Comparisons like `hands == 7` give a grid of `True` and `False`, and `.sum()` counts the `True` values.

### Reading error messages

An error isn't a failure. It tells you what went wrong. Read the **last line** first, since it names the problem (`NameError`, `KeyError`, `FileNotFoundError`), and the lines above it show where. Most errors are a typo, a missing quote, or a cell that wasn't run yet.

### Practice

Do these in your notebook.

1. Make a list of five numbers. Print the biggest one with `max(...)` and the average with `sum(...) / len(...)`.
2. Write a function `is_hot(temp)` that returns `True` when `temp` is above 30.
3. In the DataFrame above, how many rows have `output` below 450? (Filter the rows, then use `.shape`.)
4. Make a dictionary with two keys of your choosing and loop over it.
5. Using the `hands` array, count how many cards in each row are greater than 6.
6. Use a loop over `range(1, 14)` to print, for each rank, how many times it appears in the whole `hands` array.

### Answers

Try the practice questions first. Then compare. Other answers are fine if they give the same results.

**1. Biggest number and average of a list**

```python
numbers = [4, 8, 15, 16, 23]
print(max(numbers))
print(sum(numbers) / len(numbers))
```

Output: `23` and `13.2`. A common slip is dividing by a typed-in `5` instead of `len(numbers)`, which breaks when the list changes.

**2. `is_hot`**

```python
def is_hot(temp):
    return temp > 30

print(is_hot(35))   # True
print(is_hot(30))   # False
```

`temp > 30` is already `True` or `False`, so no `if` is needed. Watch for `>=`, which would call exactly 30 hot.

**3. Rows with `output` below 450**

```python
print(df[df["output"] < 450].shape)
```

Output: `(2, 2)`, so **2 rows** (the 445 and the 430). The first number of `.shape` is the row count. If you printed the filtered table instead, you should see two rows.

**4. Looping over a dictionary**

```python
ages = {"Sam": 20, "Ana": 22}
for name, age in ages.items():
    print(name, age)
```

Any two keys work. The common mistake is looping over `ages` without `.items()`, which gives only the keys.

**5. Cards greater than 6 in each row**

```python
print((hands > 6).sum(axis=1))
```

Output: `[4 2]`. The first hand has 7, 7, 9, 13 and the second has 7, 8. Using `axis=0` instead counts down each column and gives five numbers. If you got that, check the `axis` note in the Arrays section.

**6. How often each rank appears in the whole array**

```python
for k in range(1, 14):
    print(k, (hands == k).sum())
```

This prints 13 lines. Rank 7 appears 3 times, ranks 3, 4, 5, 6, 8, 9 and 13 appear once each, and all others are 0. Remember `range(1, 14)` stops at 13. This is the same idea as the rank-counts building block in Week 03.


---

## Step 2: Load the Training Data

Make sure `(.venv)` is showing in your terminal and Jupyter Notebook is running (see Week 00). In the browser, open the `notebooks` folder, click **New → Python 3 (ipykernel)**, and name the notebook `01_eda`.

Our file name has **no header row**. The first line is already a hand so we need to tell pandas the column names ourselves.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "hand"]

train = pd.read_csv("../data/raw/poker-hand-training-true.data", header=None, names=cols)
train.head()
```

`S1` is the suit of card 1 and `C1` is its rank. (The dataset calls ranks "C" for card.)

Check the basics and answer in writing:

```python
train.shape
train.dtypes
train.isna().sum().sum()
train.duplicated().sum()
train.describe()
```

1. How many rows and columns are there?
2. Any missing values or duplicates?
3. Do the suits and ranks stay between 1 and 4, and all ranks between 1 abd 13?
4. duplicated() only finds rows that match in **same order**. Could the same hand show up twice with the cards in a different order? Why would duplicated() miss it?

---

## Step 3: Class Distribution

```python
counts = train["hand"].value_counts().sort_index()
print(counts)
print((counts / len(train) * 100).round(2))
```

Plot it twice, with a normal axis and with a log axis:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts.plot(kind="bar", ax=axes[0], title="Hands per class")
counts.plot(kind="bar", ax=axes[1], logy=True, title="Hands per class (log scale)")
plt.show()
```

**Questions to answer in writing:**

1. Which class is biggest? What percentage of hands is it?
2. Which classes are rarest? How many examples do they have?
3. Why does the log-scale plot show more than the normal one?
4. **Imagine a model that ignores the cards and always predicts the most common class.** About what percentage would it get right? Is that impressive?

You've just found the most important fact about this dataset. Remember the answer to question 4.

---

## Step 4: Learn the Rules

Before modeling, make sure you understand the problem yourself. In a Markdown cell, **write the rules for each hand in your own words** (don't just copy the table), and answer:

1. Does the order of the cards matter in real poker? Does it matter in this dataset's columns? (Hint: look at the first few rows. Are the cards sorted?)
2. How many different orderings can the same five cards appear in?
3. A hand can match several descriptions at once. How should a program decide which label to give?
4. In the rank numbers, what does an Ace-high straight (10, J, Q, K, A) look like? What about an Ace-low straight (A, 2, 3, 4, 5)?

Then look at real examples. Print a few rows from each class and check that you agree with the label:

```python
for h in range(10):
    print("Class", h)
    print(train[train["hand"] == h].head(2).to_string(index=False))
```

---

## Step 5: Look at the Raw Features

```python
train[["C1", "C2", "C3", "C4", "C5"]].apply(pd.Series.value_counts).plot(kind="bar", figsize=(10, 4))
plt.title("How often each rank appears in each card position")
plt.show()
```

Do the same for the suits. Are suits and ranks spread evenly? Does position matter (is card 1 different from card 5)?

**Think about this:** a model sees ten numbers. To know a hand is a flush, it has to notice that five *separate* columns all hold the same value. To know it's a pair, it has to notice two *different* rank columns match. Do you think a simple model can do that from the raw numbers? You'll test this next week.

---

## Deliverable

`01_eda.ipynb`, cleaned up so it runs from top to bottom, containing:

- [ ] The training data loaded with named columns and basic checks
- [ ] The class distribution on a normal and a log scale
- [ ] The poker hand rules written in your own words
- [ ] A few example hands from each class
- [ ] A short written summary (6-10 sentences): the class imbalance, why accuracy would be misleading, and what you expect to be hard

Before you share your work, click **Kernel → Restart Kernel and Run All Cells**.

---

## Common Mistakes

- **Forgetting `header=None`.** Pandas will treat your first row as column names and drop a hand.
- **Using the testing file now.** Leave it alone until Week 05.
- **Treating suit and rank as ordinary numbers.** Suit 4 isn't "bigger" than suit 1. Suits are labels, not amounts. Keep this in mind for Week 03.
- **Skipping the rules.** If you don't understand the hands, you can't judge your model's mistakes later.

---

## Next Week

In Week 02, you'll split the training data, build baseline models on the raw features, and see how they do on the rare hands.
