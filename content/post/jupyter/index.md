---
title: Data Mining Project - Practice Classifiers
subtitle: Real or Not? NLP with Disaster Tweets
summary: • Project based on a Kaggle Competition on Classifiers in data mining. • Created a jupyter notebook (using python) to refine given data and conclude results. • Predicted which Tweets are about real disasters and which ones are not."""
authors:
- admin
tags: []
categories: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

---
***The purpose is of this notebook to submit my project of Data Mining to practice classifier and feature engineering.****

***The reference for this code is taken from the following:**
1. https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
2. https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
3. https://www.kaggle.com/holfyuen/basic-nlp-on-disaster-tweets
---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

![png](./output_1.png)

---
## 1. Import Data
---

```python
# Import libraries required

import seaborn as sns
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data files
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sample_submission_data = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print (train_data.shape)
print (test_data.shape)
print (sample_submission_data.shape)
```

![png](./output_2.png)

---
## 2. Refine Data
---

```python
# Remove duplicate values from train_data
train_data = train_data.drop_duplicates().reset_index(drop=True)
```

```python
# Check blank values in train_data
train_data.isnull().sum()
```

![png](./output_4.png)

```python
# Check blank values in test_data
test_data.isnull().sum()
```

![png](./output_5.png)

---
We can see there are very less blank values for "keyword" and more blank values for "location" in train as well as test data.

So, we don't need to replace blank "keyword" and for blank "location" we will check later if it is very much affecting our target.
---

---
## 3. Keywords
---

```python
# Check number of unique "keywords"
print ("Train data unique keywords", train_data.keyword.nunique())
print ("Test data unique keywords", test_data.keyword.nunique())

#We can see the number of unique keywords are same for both datasets
```

![png](./output_6.png)

---
Find out the Top 20 keywords for disaster and non-disaster
---

```python
# Most common "keywords"

plt.figure(figsize=(9,6))

sns.countplot(y=train_data.keyword, order = train_data.keyword.value_counts().iloc[:25].index)
plt.title('Top 20 keywords')
plt.show()
```

![png](./output_7.png)

```python
key_d = train_data[train_data.target==1].keyword.value_counts().head(20)
key_nd = train_data[train_data.target==0].keyword.value_counts().head(20)

plt.figure(figsize=(13,5))

plt.subplot(121)
sns.barplot(key_d, key_d.index, color='green')
plt.title('Top 20 keywords for disaster tweets')

plt.subplot(122)
sns.barplot(key_nd, key_nd.index, color='red')
plt.title('Top 20 keywords for non-disaster tweets')

plt.show()
```

![png](./output_8.png)

```python
top_key_d = train_data.groupby('keyword').mean()['target'].sort_values(ascending=False).head(20)
top_key_nd = train_data.groupby('keyword').mean()['target'].sort_values().head(20)

plt.figure(figsize=(13,5))

plt.subplot(121)
sns.barplot(top_key_d, top_key_d.index, color='blue')
plt.title('Keywords with highest percentage of disaster tweets')

plt.subplot(122)
sns.barplot(top_key_nd, top_key_nd.index, color='orange')
plt.title('Keywords with lowest percentage of disaster tweets')

plt.show()
```

![png](./output_9.png)
