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

---
We didn't find any common values for top-20 disaster adn non-disaster keywords between train_data and test_data

---

---
## 4. Locations
---

```python
# Check number of unique "locations"
print ("Train data unique locations", train_data.location.nunique())
print ("Test data unique locations", test_data.location.nunique())

# We can see the number of unique locations are not same for both datasets
```

![png](./output_10.png)

```python
# Top 20 Locations

plt.figure(figsize=(9,6))
sns.countplot(y=train_data.location, order = train_data.location.value_counts().iloc[:20].index)

plt.title('Top 20 locations')
plt.show()
```

![png](./output_11.png)

---
As we can see there are same locations with different names like "USA" and "United States", we can merge them to get cleaner data.
Also we need to check percentage of disaster tweets for these locations.

---

```python
raw_loc = train_data.location.value_counts()
top_loc = list(raw_loc[raw_loc>=10].index)
top_only = train_data[train_data.location.isin(top_loc)]

top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)
plt.figure(figsize=(14,6))
sns.barplot(x=top_l.index, y=top_l)
plt.axhline(np.mean(train_data.target))
plt.xticks(rotation=80)
plt.show()
```

![png](./output_12.png)

---
## 5. Clean missing and duplicate values

As we discussed earlier, we need to clean data for further processing as it is going to have a big effect on our resultant target values.

Also, we need to merge the locations with same meaning.
for eg. "USA" and "United States", "New York City" and "NYC", etc

---

```python
# Re-fill missing values
for col in ['keyword','location']:
    train_data[col] = train_data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')


# Merge locations with same meaning
def clean_loc(x):
    if x == 'None':
        return 'None'
    elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':
        return 'World'
    elif 'New York' in x or 'NYC' in x:
        return 'New York'    
    elif 'London' in x:
        return 'London'
    elif 'Mumbai' in x:
        return 'Mumbai'
    elif 'Washington' in x and 'D' in x and 'C' in x:
        return 'Washington DC'
    elif 'San Francisco' in x:
        return 'San Francisco'
    elif 'Los Angeles' in x:
        return 'Los Angeles'
    elif 'Seattle' in x:
        return 'Seattle'
    elif 'Chicago' in x:
        return 'Chicago'
    elif 'Toronto' in x:
        return 'Toronto'
    elif 'Sacramento' in x:
        return 'Sacramento'
    elif 'Atlanta' in x:
        return 'Atlanta'
    elif 'California' in x:
        return 'California'
    elif 'Florida' in x:
        return 'Florida'
    elif 'Texas' in x:
        return 'Texas'
    elif 'United States' in x or 'USA' in x:
        return 'USA'
    elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:
        return 'UK'
    elif 'Canada' in x:
        return 'Canada'
    elif 'India' in x:
        return 'India'
    elif 'Kenya' in x:
        return 'Kenya'
    elif 'Nigeria' in x:
        return 'Nigeria'
    elif 'Australia' in x:
        return 'Australia'
    elif 'Indonesia' in x:
        return 'Indonesia'
    elif x in top_loc:
        return x
    else: return 'Others'
    
train_data['location_clean'] = train_data['location'].apply(lambda x: clean_loc(str(x)))
test_data['location_clean'] = test_data['location'].apply(lambda x: clean_loc(str(x)))
```

```python
# Re-check cleaned value for train_data
print(train_data)
```

![png](./output_14.png)
