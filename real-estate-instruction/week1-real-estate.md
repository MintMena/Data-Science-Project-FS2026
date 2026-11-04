# Week 01: Explore the Data

**Project:** Real Estate : Predicting House Prices 
**Notebook:** `notebooks/01_eda.ipynb`

This week we'll take a look at the dataset!

We won't build any model yet. Good model starts with knowing your data well. 
Everything you notice in the data will help you make better choices and predictions.

**By the end of this week you will be able to:**

- Load a CSV file into pandas and check what's inside
- Look for missing values, duplicates and out-of-batch values
- Make histograms, scatter plots and correlation heatmap, and explain what they show in plain words

Expected time for content ~ 1.5 hour

---

## The problem

House prices in Sindian District (New Taipei City, Taiwan) are all over the place. 
Over the next weeks, you'll build a model that predicts a house's **price per area** from 6 facts about it

| Column in the file | Our short name | What it means | Unit |
|:---|:---|:---|:---|
| X1 transaction date | `trans_date` | when the house sold (2013.250 = March 2013) | year |
| X2 house age | `house_age` | how old the house is | years |
| X3 distance to the nearest MRT station | `dist_mrt` | distance to the closest metro station | metres |
| X4 number of convenience stores | `n_stores` | shops within walking distance | count |
| X5 latitude | `lat` | north-south position | degrees |
| X6 longitude | `lon` | east-west position | degrees |
| Y house price of unit area | `price` | **what we predict** | 10,000 NTD per Ping |

A **Ping** is a Taiwanese unit of area, about ~ 3.3 m^2.
**NTD** is the New Taiwan Dollar.
So, a price of 40 means 400,000 NTD per Ping.

Because price is a number and not a category, this is a **regression** problem.

---

## Step 1 : Get the Data

You don't need to search for anything. We already prepared the file for you.

1. Download the dataset real_estate_train.csv (github link -> real-estate/real_estate_train.csv)
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

Open a terminal, go to your project folder, activate .venv and start Jupyter (see Week 00). 
In the browser, open the `notebooks` folder, click **New → Python 3 (ipykernel)**, and name the notebook `01_eda`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/real_estate_train.csv")
df.head()
```
The original column names are long and have spaces, annoying to type. 
The 'No' column is just a row number, so it tells us nothing about price. Drop it and rename the rest.

```python
df = df.drop(columns = "No")
df.columns = ["trans_date", "house_age", "dist_mrt", "n_stores", "lat", "lon", "price"]
```
Now run each of these in its own cell :

```python
df.shape              # (rows, columns)
df.dtypes             # type of each column
df.isna().sum()       # missing values in each column
df.duplicated().sum() # rows that are exact copies
df.describe()         # min, max, mean, percentiles...
```
In describe(), 25%, 50% and 75% rows are percentiles. For example, 75% means 3 out of 4 houses are at or below that value.

Answer in Markdown cell:
1. How many rows and columns are there?
2. Any missing values or duplicate rows?
3. Look at the min and max of each column. Do they make sense for the unit?
  (Can a house be 0 years old? Can distance be 6,000 metres?)
4. Look at 'price'. Is the max far away from 75th percentile? What could that mean?

---

## Step 3 : Look at Each Column

A histogram shows how the values of one column are spread out.

```python
df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig("../reports/figures/histograms.png")
plt.show()
```
 Answer these in markdown :
1. Which columns look roughly like a bell shape?
2. Which ones are lopsided, with long tail on one side?
3. Does any single value look like it doesn't belong there? An outlier?

---

## Step 4 : Look at Relationship With Price

First, plot each feature againt price :

```python
features = ["trans_date", "house_age", "dist_mrt", "n_stores", "lat", "lon"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, features):
    ax.scatter(df[col], df["price"], s=10, alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel("price")
plt.tight_layout()
plt.savefig("../reports/figures/scatter_vs_price.png")
plt.show()
```
Next, a correlation heatmap.
Correlation ('r') is a number from -1 to 1 where close to 1 means positive correlation 
(one goes up, the othe one also goes up) and -1 means negative correlation (one goes up, the other goes down).
Close to 0 means no straight-line relationship

```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.savefig("../reports/figures/correlation.png")
plt.show()
```
  > Dark red = strong positive,
  > Dark blue = strong negative,
  > Pale = weak correlation

Last but not least, a map. Each dot is one house, placed by its longitude and latitude, coloured by price:


```python
plt.figure(figsize=(7, 6))
plt.scatter(df["lon"], df["lat"], c=df["price"].clip(upper=80), cmap="viridis", s=15)
plt.colorbar(label="price")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Houses coloured by price")
plt.savefig("../reports/figures/price_map.png")
plt.show()
```
Answer these in markdown :
1. Which feature has the strongest relationship with price? Positive or negative? Does it make sense in real life?
2. Which feature has the weakest relationship?
3. Is the relationship between 'dist_mrt' and price a straight line or a curve?
4. Look at the 'house_age' plot. Are the newest houses the most expensive? What about the oldest one?
5. Where on the map are the expensive houses?

> A correlation number only measures **straight-line relationships**.
> A scatter plot can show you a curve that the number misses. Always look at both.
>
> ---
>
> ---

## Deliverable

`01_eda.ipynb`, cleaned up so it runs from top to bottom, containing:

- [ ] The data is loaded, renamed and checked (shape, types, missing values, duplicates, summary)
- [ ] Histograms of every column
- [ ] Scatter plots of each feature against price
- [ ] The correlation heatmap and the price map
- [ ] Your main plots saved in reports/figures/ and short summary of what you found

Before you share your work, click **Kernel → Restart Kernel and Run All Cells**.

---

## Common Mistakes

- **Editing the raw file`.** Rename columns in pandas and keep 'data/raw/' untouched
- **Trusting the correlation number alone.** It misses curves, like one in 'dist_mrt' and the U-shape in house_age. Check scatter plots.

---

## Next Week

In Week 02, you'll take a closer look at suspicious 117.5 house and build new feature columns tha make patterns easier for a model to see.





