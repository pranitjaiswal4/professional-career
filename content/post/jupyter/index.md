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

![png](./output_0.png)

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

```python
# Check blank values in test_data
test_data.isnull().sum()
```
