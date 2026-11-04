# Week 01: Explore the Data

**Project:** Predicting Heart Failure Survival
**Notebook:** `notebooks/01_eda.ipynb`

This week we'll take a look at the dataset! First, we want to know the patients
1. What was measured
2. What normal values look like
3. Which measurements are different between patients who died and patients who survived. 

This is also where you'll meet the one column we are **never** allowed to use for predicting so take your time exploring dataset.

**By the end of this week you will be able to:**

- Load a CSV file into pandas and check what's inside
- Compare two groups of patients with box plots and death rates
- Use a statistical test and explain what p-value means

Expected time for content ~ 1.5 hour

---

## The problem

These are medical records of patients with heart failure, collected at a hospital in Faisalabad, Pakistan, 2015.
Every patient was followed for some time after their visit.

Over the next week, you'll build a model that **predicts whether a patient died during that follow-up period**.

| Column | What it means | Type / Unit |
|:---|:---|:---|
| `age` | age of the patient | years |
| `anaemia` | low red blood cells or haemoglobin | yes/no (1/0) |
| `creatinine_phosphokinase` | level of the CPK enzyme in the blood | mcg/L |
| `diabetes` | patient has diabetes | yes/no |
| `ejection_fraction` | % of blood the heart pumps out with each beat | % |
| `high_blood_pressure` | patient has hypertension | yes/no |
| `platelets` | platelets in the blood | kiloplatelets/mL |
| `serum_creatinine` | creatinine in the blood (a sign of how well the kidneys work) | mg/dL |
| `serum_sodium` | sodium in the blood | mEq/L |
| `sex` | 1 = man, 0 = woman | yes/no |
| `smoking` | patient smokes | yes/no |
| `time` | how many days the patient was followed | days |
| `DEATH_EVENT` | **what we predict:** 1 = died during follow-up | yes/no |

Because the target is a category (died or survived), this is a **classification** problem.

**The time rule**
'time' is how long each patient was followed.
Think about patient who died on day 10, they can only be followed for 10 days so a short 'time' almost gives away the answer.
Using a column that secretly contains the answer is called **data leakage**. Realistically, a doctor seeing new patient would not know this number yet.

You'll look at 'time' this week and use properly in Week 02 but **never use it as a model input**
> Disclaimer : this is a leaning project, not medical tool so nothing here should be used to make decisions about real patients

---

## Step 1: Get the Data
You don't need to search for anything. We already prepared the file for you.

1. Download the dataset heart_failure_train.csv (github link -> heart-failure/heart_failure_train.csv)
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

## Step 2 : Load and Check

ake sure `(.venv)` is showing in your terminal and Jupyter Notebook is running (see Week 00). 
In the browser, open the `notebooks` folder, click **New → Python 3 (ipykernel)**, and name the notebook `01_eda`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/heart_failure_train.csv")
df.head()
```
Now run each of these in its own cells :

```python
df.shape                          # (rows, columns)
df.isna().sum()                   # missing values in each column
df.duplicated().sum()             # rows that are exact copies
df.describe()                     # min, max, mean, percentiles...
df["DEATH_EVENT"].value_counts()  # how many died (1) and survived (0)
```
Answer in markdown cell :
1. How many patients are there? How many died, and what percent is that?
2. Which columns are yes/no (only 0 and 1), and which are real numbers?
3. Look at the min and max of each number column. Do they make sense for the unit?
   Do any look extreme? (Look at 'creatinine_phosphokinase' and 'serum_creatinine')

---

## Step 3 : Compare Patients Who Died vs Survived

**Number columns : box plots**

A box plot shows the middle half of the values as a box, the median as a line inside it, and unusual values as dots. 
Putting two boxes side by side lets you compare the groups.

```python
sns.boxplot(data=df, x="DEATH_EVENT", y="serum_creatinine")
plt.xlabel("DEATH_EVENT (0 = survived, 1 = died)")
plt.ylabel("serum creatinine (mg/dL)")
plt.show()
```
**Your turn** : write a loop that draws one box plot for each of the six number columns :
- 'age'
- 'creatinine_phosphokinase'
- 'ejection_fraction'
- 'platelets'
- 'serum_creatinine'
- 'serum_sodium'

> **Hint**
> 1. Put the six column names in a list.
> 2. Make a grid with
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
```
> 3. Loop over zip(axes.flat, your_list) and call sns.boxplot(...) with ax=ax inside the loop.
> 4. Save it to 'reports/figures/' before plt.show()

**Yes/No columns : death rates**

For a yes/no column, the easiest comparison is the death rate in each group. 
Because 'DEATH_EVENT' is 0 or 1, its mean is the share of patients who died.

```python
df.groupby("smoking")["DEATH_EVENT"].mean()
```
Do the same for 'anaemia', 'diabetes', 'high_blood_pressure' and 'sex'.

---

## Step 4 : Is the Difference Real?

A box plot can show a difference, but with only 209 patients, some differences happen just by chance. 
A statistical test helps you decide.

The test gives a **p-value**.
A small p-value means "a difference this big would be unlikely if there were really no difference". 
We usually call p below 0.05 (p <= 0.05) **significant**.

| Column type | Test | Code |
|:---|:---|:---|
| `numbers` | Mann-Whitney U | stats.mannwhitneyu(died_values, survived_values).pvalue |
| `yes/no` | chi-square | stats.chi2_contingency(pd.crosstab(df[col],df["DEATH_EVENT"]))[1] |

Here's how to get the 2 groups for 1 number column.

```python
from scipy import stats
# stats library contains pvalue function that we can use

died = df[df["DEATH_EVENT"] == 1]
survived = df[df["DEATH_EVENT"] == 0]

stats.mannwhitneyu(died["age"], survived["age"]).pvalue
```
**Your turn** : run the right test for every column except 'DEATH_EVENT'. 
Save the results in a dictionary ({"age": 0.0004, ...}), turn it into a table with pd.Series(...), and sort it from smallest to largest p-value. 
Include 'time' here. Looking at it is fine, just never use it in a model.

Answer these in markdown cells :
1. Which columns are significant (p below 0.05)?
2. Were you surprised that smoking and diabetes are **not** significant here? What could explain it?
3. Describe the patients who died in plain words.

---

## Deliverable

`01_eda.ipynb`, cleaned up so it runs from top to bottom, containing:

- [ ] The data loaded and checked (shape, missing values, duplicates, summary, target counts)
- [ ] Box plots for 6 number columns and death rates for yes/no column
- [ ] A sorted table of p-values
- [ ] Your main plots saved in reports/figures/
- [ ] A short written summary (6-10 sentences): the class imbalance, why accuracy would be misleading, and what you expect to be hard

Before you share your work, click **Kernel → Restart Kernel and Run All Cells**.

---

## Common Mistakes

- **Significant ≠ Big** : Significance means "probably not just a chance".  A tiny difference can be significant and big-looking one may not be.
- **Treating 'time' as normal feature** : Never predict with what you collect in the future!
---

## Next Week

In Week 02, you'll draw survival curves, see when most deaths happen and learn exactly why 'time' is a leakage.
