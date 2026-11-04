# Week 01: Explore the Data

**Project:** Predicting Power Plant Output
**Notebook:** `notebooks/01_eda.ipynb`

This week you get the dataset and learn what's in it before you build anything. Good models start with understanding the data.

**By the end of this week you will be able to:**

- Load a dataset into pandas and inspect it
- Check for missing values, duplicates, and odd values
- Make histograms and scatter plots, and read a correlation heatmap
- Describe in plain words how each variable relates to power output

Expected time for content ~ 1 hour

---

## The Problem

A combined cycle power plant makes electricity with gas turbines and steam turbines. How much power it produces each hour depends on the conditions around it. Your goal over the next six weeks is to **predict the plant's hourly electrical output from four measurements**.

| Column | Meaning | Unit |
|---|---|---|
| `AT` | Ambient temperature | °C |
| `V` | Exhaust vacuum | cm Hg |
| `AP` | Ambient pressure | millibar |
| `RH` | Relative humidity | % |
| `PE` | Net hourly electrical energy output (**the target**) | MW |

**MW** stands for megawatt, a unit of power. One megawatt is one million watts.

Because `PE` is a number, not a category, this is a **regression** problem.

---

## Step 1: Get the Data
You don't need to search for anything. We already prepared the file for you.

1. Download the dataset power-plant.csv (github link -> power-plant/power_plant_train.csv)
   https://github.com/MintMena/Data-Science-Project-Dataset-FS2026
   > On Github, click the file name then click download button ("Download raw file") on top        right of the file view
2. Move the file into `data/raw/` folder inside your poker-hand-project folder from Week 00
> **Never edit files in `data/raw/**

> To check that it's in there.

   > Windows : terminal -> dir data\raw

   > Mac/Linux : terminal -> ls data/raw

The exact file names may differ a little. Use the names you actually see. Open each file in a text editor if you like: if the first line is already a hand, there is no header row.

---

## Warm-Up: The Python You'll Need

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

### Reading error messages

An error isn't a failure. It tells you what went wrong. Read the **last line** first, since it names the problem (`NameError`, `KeyError`, `FileNotFoundError`), and the lines above it show where. Most errors are a typo, a missing quote, or a cell that wasn't run yet.

### Practice

Do these in your notebook.

1. Make a list of five numbers. Print the biggest one with `max(...)` and the average with `sum(...) / len(...)`.
2. Write a function `is_hot(temp)` that returns `True` when `temp` is above 30.
3. In the DataFrame above, how many rows have `output` below 450? (Filter the rows, then use `.shape`.)
4. Make a dictionary with two keys of your choosing and loop over it.

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


---

## Step 2: Load and Inspect

Make sure `(.venv)` is showing in your terminal and Jupyter Notebook is running (see Week 00). In the browser, open the `notebooks` folder, click **New → Python 3 (ipykernel)**, and name the notebook `01_eda`. Start with:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/power_plant_train.csv")
df.head()
```

> The `../` means "go up one folder from `notebooks/`". If you get a file-not-found error, check your file name and folder.

Then answer each of these in a notebook cell (use a Markdown cell for your written answers):

```python
df.shape            # rows and columns
df.dtypes           # data types
df.isna().sum()     # missing values per column
df.duplicated().sum()   # duplicate rows
df.describe()       # summary statistics
```

**Questions to answer in writing:**

1. How many rows and columns are there?
2. Are there any missing values? Any duplicate rows?
3. What are the minimum and maximum of each column? Do they look reasonable for the unit?

If you find duplicate rows, write down how many, but **keep them** in the data. These are hourly sensor readings, and two hours can have the same measurements without either one being a mistake. Removing real readings would throw away information.

---

## Step 3: Look at Each Variable

Make a histogram for each column:

```python
df.hist(bins=30, figsize=(10, 8))
plt.tight_layout()
plt.show()
```

**Questions:** Which columns look roughly bell-shaped? Which look lopsided or have more than one peak? Are there any values that look like outliers?

---

## Step 4: Look at Relationships

Plot each feature against the target:

```python
features = ["AT", "V", "AP", "RH"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, features):
    ax.scatter(df[col], df["PE"], s=3, alpha=0.3)
    ax.set_xlabel(col)
    ax.set_ylabel("PE")
plt.tight_layout()
plt.show()
```
s = 3 makes the dots small and alpha = 0.3 makes them see-through. With almost 7,000 points, that's the only way to see where most of them sit.

Then make a correlation heatmap
> Correlation ('r') is a number from -1 to 1 where close to 1 means positive correlation (if one goes up, the other one also goes up) and -1 means negative correlation (if one goes up, the other one goes down). Close to 0 means no straight-line relationship.
:

```python
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.show()
```

Dark red means a strong positive relationship, dark blue means a strong negative one, and pale means weak.

**Questions to answer in writing:**

1. Which feature has the strongest relationship with `PE`? Is it positive or negative? Does that make sense physically?
2. Which feature has the weakest relationship?
3. Which two **features** are strongly correlated with each other? (Hint: look at `AT` and `V`.) Why might that matter for a model?
4. Are any relationships curved rather than a straight line?

A correlation number only measures straight-line relationships, and scatter plots can show curves it misses. Use both.

---

## Deliverable

`01_eda.ipynb`, cleaned up so it runs from top to bottom, containing:

- [ ] The data loaded and inspected (shape, types, missing values, duplicates, summary)
- [ ] Histograms of every column
- [ ] Scatter plots of each feature against `PE`
- [ ] A correlation heatmap
- [ ] A short written summary (5-8 sentences) of what you found, in your own words

Before you share your work, click **Kernel → Restart Kernel and Run All Cells** to make sure everything runs in order.

---

## Common Mistakes

- **Editing the raw file.** Don't. Keep `data/raw/` untouched.
- **Describing instead of explaining.** "AT and PE are correlated" is a start. "As temperature rises, output falls, probably because..." is better.
- **Ignoring units.** A number means little without its unit.
- **Forgetting Run All.** A notebook that only works if cells run in a certain order is a broken notebook.

---

## Next Week

In Week 02, you'll split the data into training and test sets and build your first model: a baseline to beat.
