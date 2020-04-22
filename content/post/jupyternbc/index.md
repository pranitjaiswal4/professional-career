---
title: Data Mining Assignment - Learn about Naive Bayes Classifier (NBC)
subtitle: Implementation of Naive Bayes Classifier and Effect of Smoothing
summary:
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
## Download Links:
<a href="https://pranitjaiswal.netlify.com/files/Jaiswal_03.ipynb" download>Jupyter Notebook (.ipynb)</a>   

<a href="https://drive.google.com/open?id=1N6xowmiSTo2LAIIp7qnQu9pV4ryyJsVe" download>Input Data (.tar.gz)</a>

---

## a. Divide the dataset as train, development, and test


```python
# imports here
import numpy as np
import pandas as pd
import tarfile
import matplotlib.pyplot as plt
import scipy as sp
import csv
import random
import math
import operator
import os
from collections import Counter
import nltk
```


```python
# Extract and Combine data from the given dataset (.tar.gz file)
# Note: This will take a long time to execute as the data is in distinct files and very large

df_train = pd.DataFrame()
df_dev_test = pd.DataFrame()
df_dev = pd.DataFrame()
df_test = pd.DataFrame()

labels = {'pos': 1, 'neg': 0}

tar = tarfile.open("aclImdb_v1.tar.gz", "r:gz")
for member in tar.getmembers():
    f = tar.extractfile(member)
    if f is not None:
        if(member.name.find('aclImdb/train/pos/') != -1):
            content = f.read().decode('utf-8')
            df_train = df_train.append([[content, labels['pos']]],
                           ignore_index=True)
            
        if(member.name.find('aclImdb/train/neg/') != -1):
            content = f.read().decode('utf-8')
            df_train = df_train.append([[content, labels['neg']]],
                           ignore_index=True)
            
        if(member.name.find('aclImdb/test/pos/') != -1):
            content = f.read().decode('utf-8')
            df_dev_test = df_dev_test.append([[content, labels['pos']]],
                           ignore_index=True)
            
        if(member.name.find('aclImdb/test/neg/') != -1):
            content = f.read().decode('utf-8')
            df_dev_test = df_dev_test.append([[content, labels['neg']]],
                           ignore_index=True)
            
df_train.columns = ['review', 'sentiment']
df_dev_test.columns = ['review', 'sentiment']
```


```python
# Split dataframes in test folder randomly into dev and test dataframes

spdf = np.random.rand(len(df_dev_test)) < 0.6

df_dev = df_dev_test[spdf]
df_test = df_dev_test[~spdf]
```


```python
# Print information regarding records in each dataset

df_train_len = len(df_train)
print('total train records:', df_train_len)

df_train_pos_len = len(df_train[df_train['sentiment'] == 1])
df_train_neg_len = len(df_train[df_train['sentiment'] == 0])
prob_pos_train = df_train_pos_len / df_train_len
prob_neg_train = df_train_neg_len / df_train_len

print ('positive records:', df_train_pos_len)
print ('negative records:', df_train_neg_len)

print ('prob positive records:', prob_pos_train)
print ('prob negative records:', prob_neg_train)

print()
print('total dev records:', len(df_dev))
print('total test records:', len(df_test))
```

    total train records: 25000
    positive records: 12500
    negative records: 12500
    prob positive records: 0.5
    prob negative records: 0.5
    
    total dev records: 15059
    total test records: 9941
    


```python
# Print count of each sentiment in train dataset
df_train.groupby('sentiment').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
    </tr>
    <tr>
      <th>sentiment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print count of each sentiment in dev dataset
df_dev.groupby('sentiment').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
    </tr>
    <tr>
      <th>sentiment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7443</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7616</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print count of each sentiment in test dataset
df_test.groupby('sentiment').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
    </tr>
    <tr>
      <th>sentiment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5057</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4884</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create CSV files for each dataframe

np.random.seed(0)
df_train = df_train.reindex(np.random.permutation(df_train.index))
df_train.to_csv('movie_data_train.csv', index=False, encoding = 'utf-8')

np.random.seed(0)
df_dev = df_dev.reindex(np.random.permutation(df_dev.index))
df_dev.to_csv('movie_data_dev.csv', index=False, encoding = 'utf-8')

np.random.seed(0)
df_test = df_test.reindex(np.random.permutation(df_test.index))
df_test.to_csv('movie_data_test.csv', index=False, encoding = 'utf-8')
```


```python
# Print train data
df_train = pd.read_csv('movie_data_train.csv', encoding = 'utf-8')
df_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Fräulein Doktor is as good a demonstration as ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>I watched this knowing almost nothing about it...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>I must give How She Move a near-perfect rating...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>The storyline is absurd and lame,also sucking ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>I watched Grendel the other night and am compe...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print dev_data
df_dev = pd.read_csv('movie_data_dev.csv', encoding = 'utf-8')
df_dev.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Scary Movie 1-4, Epic Movie, Date Movie, Meet ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>This is a funny, intelligent and, in a sense, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>I give this movie 2 stars purely because of it...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>When at the very start of the film Paleontolog...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>I saw this movie awhile back and can't seem to...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print test data
df_test = pd.read_csv('movie_data_test.csv', encoding = 'utf-8')
df_test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>I've been a fan of Larry King's show for awhil...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>THE FEELING of the need to have someone play t...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>"Bride of Chucky" is one of the better horror ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>I just purchased this movie because I love to ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>This film is great - well written and very ent...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate length of training dataset

doc_len = len(df_train)
print(doc_len)
```

    25000
    


```python
# Replace all the possible special characters to get proper words

df_train.columns = df_train.columns.str.strip()         
df_train.columns = df_train.columns.str.replace(r"[^a-zA-Z\d\_]+", "")    
df_train.columns = df_train.columns.str.replace(r"[^a-zA-Z\d\_]+", "")

df_dev.columns = df_dev.columns.str.strip()         
df_dev.columns = df_dev.columns.str.replace(r"[^a-zA-Z\d\_]+", "")    
df_dev.columns = df_dev.columns.str.replace(r"[^a-zA-Z\d\_]+", "")

df_test.columns = df_test.columns.str.strip()         
df_test.columns = df_test.columns.str.replace(r"[^a-zA-Z\d\_]+", "")    
df_test.columns = df_test.columns.str.replace(r"[^a-zA-Z\d\_]+", "")

df_train = df_train.replace([";",":","=","\+","<", ">", "\?", "!", "\\\\", "@", "#", "$", "\*", "%", ",", "\.", "\(", "\)", "\[", "\]", "\{", "\}", "\"", "/br"], "", regex = True)
df_dev = df_dev.replace([";",":","=","\+","<", ">", "\?", "!", "\\\\", "@", "#", "$", "\*", "%", ",", "\.", "\(", "\)", "\[", "\]", "\{", "\}", "\"", "/br"], "", regex = True)
df_test = df_test.replace([";",":","=","\+","<", ">", "\?", "!", "\\\\", "@", "#", "$", "\*", "%", ",", "\.", "\(", "\)", "\[", "\]", "\{", "\}", "\"", "/br"], "", regex = True)

df_train = df_train.replace(["' ", " '"], " ", regex = True)
df_dev = df_dev.replace(["' ", " '"], " ", regex = True)
df_test = df_test.replace(["' ", " '"], " ", regex = True)
```

## b. Build a vocabulary as list


```python
# Calculate frequency for each vocab

wordfreq = dict()
wordfreq_pos = dict()
wordfreq_neg = dict()
for ind in df_train.index:
    review_set = set(df_train['review'][ind].lower().split())
    for word in review_set:
        if word in wordfreq:
            wordfreq[word] += 1
        else:
            wordfreq[word] = 1
        
        if df_train['sentiment'][ind] == 1:
            if word in wordfreq_pos:
                wordfreq_pos[word] += 1
            else:
                wordfreq_pos[word] = 1
        else:
            if word in wordfreq_neg:
                wordfreq_neg[word] += 1
            else:
                wordfreq_neg[word] = 1
```


```python
# Ommiting rare vocabs having frequency < 5

final_vocab = dict()
final_vocab_pos = dict()
final_vocab_neg = dict()
for word in wordfreq:
    if wordfreq[word] > 5:
        final_vocab[word] = wordfreq[word]
    if word in wordfreq_pos:
        if wordfreq_pos[word] > 5:
            final_vocab_pos[word] = wordfreq_pos[word]
    if word in wordfreq_neg:
        if wordfreq_neg[word] > 5:
            final_vocab_neg[word] = wordfreq_neg[word]
```


```python
# Print count of filtered vocab
print("Total Vocab:",len(final_vocab))
```

    Total Vocab: 27681
    

## c. Calculate the probabilities


```python
# Calculate probabilities for:
# a) each word
# b) each word given positive
# c) each word given negative

prob_word = dict()
prob_word_g_pos = dict()
prob_word_g_neg = dict()

for word in final_vocab:
    prob_word[word] = final_vocab[word] / doc_len
    if word in final_vocab_pos:
        prob_word_g_pos[word] = final_vocab_pos[word] / df_train_pos_len
        
    if word in final_vocab_neg:
        prob_word_g_neg[word] = final_vocab_neg[word] / df_train_neg_len
```

## d. Calculate accuracy using dev dataset


```python
# Predict values for dev dataset using prob(pos|all_words)
accuracy_normal = []
df_dev_arr = np.array_split(df_dev, 5)
ctr = 0

print("Accuracy using 5-Fold Cross Validation:")

for df in df_dev_arr:
    count = 0
    ctr += 1
    predicted_sentiments = []
    prob_pos_g_wir = dict()
    prob_neg_g_wir = dict()
    
    for ind in df.index:
        numPos = 0.00
        numNeg = 0.00
        
        review_set = set(df['review'][ind].lower().split())
        for word in review_set:
            if word in prob_word:
                if word not in prob_word_g_pos:
                    numNeg = 0
                elif word not in prob_word_g_neg:
                    numPos = 0
                else:
                    numPos = numPos + math.log(prob_word_g_pos[word])
                    numNeg = numNeg + math.log(prob_word_g_neg[word])
                            
        prob_pos_g_wir[ind] = pow(math.e, numPos) * prob_pos_train
        prob_neg_g_wir[ind] = pow(math.e, numNeg) * prob_neg_train
                            
        if(prob_pos_g_wir[ind] < prob_neg_g_wir[ind]):
            predicted_sentiments.append(0)
        else:
            predicted_sentiments.append(1)
                                    
    df['prediction'] = predicted_sentiments
                                                                        
    for ind in df.index:
        if df['sentiment'][ind] == df['prediction'][ind]:
            count += 1
                                            
    accuracy = count / len(df)
    accuracy_normal.append(accuracy)
    print (ctr,": Accuracy df_dev:",accuracy*100,"%")
```

    Accuracy using 5-Fold Cross Validation:
    1 : Accuracy df_dev: 61.254980079681275 %
    2 : Accuracy df_dev: 61.75298804780876 %
    3 : Accuracy df_dev: 63.081009296148736 %
    4 : Accuracy df_dev: 62.35059760956175 %
    5 : Accuracy df_dev: 62.570574559946856 %
    

## e. Perform given experiments

#### e.1 Compare the effect of Smoothing


```python
# Calculate probabilities using Smoothing for:
# a) each word given positive
# b) each word given negative

prob_word_g_pos_smooth = dict()
prob_word_g_neg_smooth = dict()

for word in final_vocab:
    if word in final_vocab_pos:
        prob_word_g_pos_smooth[word] = (final_vocab_pos[word]+1) / (df_train_pos_len + len(final_vocab))
        
    if word in final_vocab_neg:
        prob_word_g_neg_smooth[word] = (final_vocab_neg[word]+1) / (df_train_neg_len + len(final_vocab))
```


```python
# Predict values for dev dataset using prob(pos|all_words)

accuracy_smooth = []
df_dev_arr = np.array_split(df_dev, 5)
ctr = 0

print("Accuracy after Smoothing using 5-Fold Cross Validation:")

for df in df_dev_arr:
    count = 0
    ctr += 1
    predicted_sentiments = []
    prob_pos_g_wir = dict()
    prob_neg_g_wir = dict()
    
    for ind in df.index:
        numPos = 0.00
        numNeg = 0.00
        
        review_set = set(df['review'][ind].lower().split())
        for word in review_set:
            if word in prob_word:
                if word not in prob_word_g_pos:
                    numNeg = 0
                elif word not in prob_word_g_neg:
                    numPos = 0
                else:
                    numPos = numPos + math.log(prob_word_g_pos_smooth[word])
                    numNeg = numNeg + math.log(prob_word_g_neg_smooth[word])
                            
        prob_pos_g_wir[ind] = pow(math.e, numPos) * prob_pos_train
        prob_neg_g_wir[ind] = pow(math.e, numNeg) * prob_neg_train
                            
        if(prob_pos_g_wir[ind] < prob_neg_g_wir[ind]):
            predicted_sentiments.append(0)
        else:
            predicted_sentiments.append(1)
                                    
    df['prediction'] = predicted_sentiments
                                                                        
    for ind in df.index:
        if df['sentiment'][ind] == df['prediction'][ind]:
            count += 1
                                            
    accuracy = count / len(df)
    accuracy_smooth.append(accuracy)
    print (ctr,": Accuracy df_dev:",accuracy*100,"%")
```

    Accuracy after Smoothing using 5-Fold Cross Validation:
    1 : Accuracy df_dev: 61.05577689243028 %
    2 : Accuracy df_dev: 61.68658698539177 %
    3 : Accuracy df_dev: 63.01460823373174 %
    4 : Accuracy df_dev: 62.21779548472776 %
    5 : Accuracy df_dev: 62.3380936565925 %
    


```python
# Compare accuracy values for normal and after smoothing

betterNormal = 0
betterSmoothing = 0

for i in range(len(accuracy_normal)):
    if accuracy_normal[i] > accuracy_smooth[i]:
        betterNormal += 1
    else:
        betterSmoothing +=1

if(betterNormal > betterSmoothing):
    print("For the given dev dataset, accuracy is better without smoothing")
else:
    print("For the given dev dataset, accuracy is better with smoothing")
```

    For the given dev dataset, accuracy is better without smoothing
    

#### e.2 Derive Top 10 words that predicts positive and negative class


```python
# Calculate prob given word for all vocab

prob_pos_given_word = dict()
prob_neg_given_word = dict()

for word in final_vocab:
    if word in final_vocab_pos:
        prob_pos_given_word[word] = (prob_word_g_pos[word] * prob_pos_train) / prob_word[word]
    if word in final_vocab_neg:
        prob_neg_given_word[word] = (prob_word_g_neg[word] * prob_neg_train) / prob_word[word]
```


```python
# Print top 10 words predicting positive class
print("Top 10 words predicting positive class:")
prob_pos_given_word = sorted(prob_pos_given_word.items(), key=operator.itemgetter(1), reverse=True)

prob_pos_given_word[:10]
```

    Top 10 words predicting positive class:
    




    [('doktor', 1.0),
     ('mccartney', 1.0),
     ('brownstone', 1.0),
     ('unwillingly', 1.0),
     ('nord', 1.0),
     ("gilliam's", 1.0),
     ('stitzer', 1.0),
     ('apatow', 1.0),
     ('edie', 1.0),
     ('shimmering', 1.0)]




```python
# Print top 10 words predicting negative class
print("Top 10 words predicting negative class:")
prob_neg_given_word = sorted(prob_neg_given_word.items(), key=operator.itemgetter(1), reverse=True)
prob_neg_given_word[:10]
```

    Top 10 words predicting negative class:
    




    [('recoil', 1.0),
     ('clowns', 1.0),
     ('unintended', 1.0),
     ('dorff', 1.0),
     ('slater', 1.0),
     ('kareena', 1.0),
     ('atari', 1.0),
     ('kargil', 1.0),
     ('weisz', 1.0),
     ('2/10', 1.0)]



## f. Calculate accuracy using test dataset


```python
# Predict values for test dataset using optimal values : prob(pos|all_words)

predicted_sentiments = []
prob_pos_g_wir = dict()
prob_neg_g_wir = dict()

if(betterNormal < betterSmoothing):
    prob_word_g_pos = prob_word_g_pos_smooth

for ind in df_test.index:
    numPos = 0.00
    numNeg = 0.00
    
    review_set = set(df_test['review'][ind].lower().split())
    for word in review_set:
        if word in prob_word:
            if word not in prob_word_g_pos:
                numNeg = 0
            elif word not in prob_word_g_neg:
                numPos = 0
            else:
                numPos = numPos + math.log(prob_word_g_pos[word])
                numNeg = numNeg + math.log(prob_word_g_neg[word])
    
    prob_pos_g_wir[ind] = pow(math.e, numPos) * prob_pos_train
    prob_neg_g_wir[ind] = pow(math.e, numNeg) * prob_neg_train
    
    if(prob_pos_g_wir[ind] < prob_neg_g_wir[ind]):
        predicted_sentiments.append(0)
    else:
        predicted_sentiments.append(1)
        
df_test['prediction'] = predicted_sentiments

count = 0

for ind in df_test.index:
    if df_test['sentiment'][ind] == df_test['prediction'][ind]:
        count += 1
        
accuracy = count / len(df_test)
print ("Accuracy df_test:",accuracy*100,"%")
```

    Accuracy df_test: 61.41233276330349 %
    
