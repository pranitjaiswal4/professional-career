---
title: Data Mining Assignment - Learn about kNN Classifier
subtitle: Implementation of kNN Classifier using Euclidean distance, Normalized Euclidean Distance, and Cosine Similarity
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

## a.	Divide the dataset as development and test


```python
# imports here
import pandas as pd
import csv
import random
import math
import operator 
```


```python
# Read CSV file and print data

filename = r'D:\Pranit\UTA\Study\DM\Assignment -2\iris_data.txt'

with open(filename) as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print (row)
```

    ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
    ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa']
    ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa']
    ['4.6', '3.1', '1.5', '0.2', 'Iris-setosa']
    ['5.0', '3.6', '1.4', '0.2', 'Iris-setosa']
    ['5.4', '3.9', '1.7', '0.4', 'Iris-setosa']
    ['4.6', '3.4', '1.4', '0.3', 'Iris-setosa']
    ['5.0', '3.4', '1.5', '0.2', 'Iris-setosa']
    ['4.4', '2.9', '1.4', '0.2', 'Iris-setosa']
    ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa']
    ['5.4', '3.7', '1.5', '0.2', 'Iris-setosa']
    ['4.8', '3.4', '1.6', '0.2', 'Iris-setosa']
    ['4.8', '3.0', '1.4', '0.1', 'Iris-setosa']
    ['4.3', '3.0', '1.1', '0.1', 'Iris-setosa']
    ['5.8', '4.0', '1.2', '0.2', 'Iris-setosa']
    ['5.7', '4.4', '1.5', '0.4', 'Iris-setosa']
    ['5.4', '3.9', '1.3', '0.4', 'Iris-setosa']
    ['5.1', '3.5', '1.4', '0.3', 'Iris-setosa']
    ['5.7', '3.8', '1.7', '0.3', 'Iris-setosa']
    ['5.1', '3.8', '1.5', '0.3', 'Iris-setosa']
    ['5.4', '3.4', '1.7', '0.2', 'Iris-setosa']
    ['5.1', '3.7', '1.5', '0.4', 'Iris-setosa']
    ['4.6', '3.6', '1.0', '0.2', 'Iris-setosa']
    ['5.1', '3.3', '1.7', '0.5', 'Iris-setosa']
    ['4.8', '3.4', '1.9', '0.2', 'Iris-setosa']
    ['5.0', '3.0', '1.6', '0.2', 'Iris-setosa']
    ['5.0', '3.4', '1.6', '0.4', 'Iris-setosa']
    ['5.2', '3.5', '1.5', '0.2', 'Iris-setosa']
    ['5.2', '3.4', '1.4', '0.2', 'Iris-setosa']
    ['4.7', '3.2', '1.6', '0.2', 'Iris-setosa']
    ['4.8', '3.1', '1.6', '0.2', 'Iris-setosa']
    ['5.4', '3.4', '1.5', '0.4', 'Iris-setosa']
    ['5.2', '4.1', '1.5', '0.1', 'Iris-setosa']
    ['5.5', '4.2', '1.4', '0.2', 'Iris-setosa']
    ['4.9', '3.1', '1.5', '0.2', 'Iris-setosa']
    ['5.0', '3.2', '1.2', '0.2', 'Iris-setosa']
    ['5.5', '3.5', '1.3', '0.2', 'Iris-setosa']
    ['4.9', '3.6', '1.4', '0.1', 'Iris-setosa']
    ['4.4', '3.0', '1.3', '0.2', 'Iris-setosa']
    ['5.1', '3.4', '1.5', '0.2', 'Iris-setosa']
    ['5.0', '3.5', '1.3', '0.3', 'Iris-setosa']
    ['4.5', '2.3', '1.3', '0.3', 'Iris-setosa']
    ['4.4', '3.2', '1.3', '0.2', 'Iris-setosa']
    ['5.0', '3.5', '1.6', '0.6', 'Iris-setosa']
    ['5.1', '3.8', '1.9', '0.4', 'Iris-setosa']
    ['4.8', '3.0', '1.4', '0.3', 'Iris-setosa']
    ['5.1', '3.8', '1.6', '0.2', 'Iris-setosa']
    ['4.6', '3.2', '1.4', '0.2', 'Iris-setosa']
    ['5.3', '3.7', '1.5', '0.2', 'Iris-setosa']
    ['5.0', '3.3', '1.4', '0.2', 'Iris-setosa']
    ['7.0', '3.2', '4.7', '1.4', 'Iris-versicolor']
    ['6.4', '3.2', '4.5', '1.5', 'Iris-versicolor']
    ['6.9', '3.1', '4.9', '1.5', 'Iris-versicolor']
    ['5.5', '2.3', '4.0', '1.3', 'Iris-versicolor']
    ['6.5', '2.8', '4.6', '1.5', 'Iris-versicolor']
    ['5.7', '2.8', '4.5', '1.3', 'Iris-versicolor']
    ['6.3', '3.3', '4.7', '1.6', 'Iris-versicolor']
    ['4.9', '2.4', '3.3', '1.0', 'Iris-versicolor']
    ['6.6', '2.9', '4.6', '1.3', 'Iris-versicolor']
    ['5.2', '2.7', '3.9', '1.4', 'Iris-versicolor']
    ['5.0', '2.0', '3.5', '1.0', 'Iris-versicolor']
    ['5.9', '3.0', '4.2', '1.5', 'Iris-versicolor']
    ['6.0', '2.2', '4.0', '1.0', 'Iris-versicolor']
    ['6.1', '2.9', '4.7', '1.4', 'Iris-versicolor']
    ['5.6', '2.9', '3.6', '1.3', 'Iris-versicolor']
    ['6.7', '3.1', '4.4', '1.4', 'Iris-versicolor']
    ['5.6', '3.0', '4.5', '1.5', 'Iris-versicolor']
    ['5.8', '2.7', '4.1', '1.0', 'Iris-versicolor']
    ['6.2', '2.2', '4.5', '1.5', 'Iris-versicolor']
    ['5.6', '2.5', '3.9', '1.1', 'Iris-versicolor']
    ['5.9', '3.2', '4.8', '1.8', 'Iris-versicolor']
    ['6.1', '2.8', '4.0', '1.3', 'Iris-versicolor']
    ['6.3', '2.5', '4.9', '1.5', 'Iris-versicolor']
    ['6.1', '2.8', '4.7', '1.2', 'Iris-versicolor']
    ['6.4', '2.9', '4.3', '1.3', 'Iris-versicolor']
    ['6.6', '3.0', '4.4', '1.4', 'Iris-versicolor']
    ['6.8', '2.8', '4.8', '1.4', 'Iris-versicolor']
    ['6.7', '3.0', '5.0', '1.7', 'Iris-versicolor']
    ['6.0', '2.9', '4.5', '1.5', 'Iris-versicolor']
    ['5.7', '2.6', '3.5', '1.0', 'Iris-versicolor']
    ['5.5', '2.4', '3.8', '1.1', 'Iris-versicolor']
    ['5.5', '2.4', '3.7', '1.0', 'Iris-versicolor']
    ['5.8', '2.7', '3.9', '1.2', 'Iris-versicolor']
    ['6.0', '2.7', '5.1', '1.6', 'Iris-versicolor']
    ['5.4', '3.0', '4.5', '1.5', 'Iris-versicolor']
    ['6.0', '3.4', '4.5', '1.6', 'Iris-versicolor']
    ['6.7', '3.1', '4.7', '1.5', 'Iris-versicolor']
    ['6.3', '2.3', '4.4', '1.3', 'Iris-versicolor']
    ['5.6', '3.0', '4.1', '1.3', 'Iris-versicolor']
    ['5.5', '2.5', '4.0', '1.3', 'Iris-versicolor']
    ['5.5', '2.6', '4.4', '1.2', 'Iris-versicolor']
    ['6.1', '3.0', '4.6', '1.4', 'Iris-versicolor']
    ['5.8', '2.6', '4.0', '1.2', 'Iris-versicolor']
    ['5.0', '2.3', '3.3', '1.0', 'Iris-versicolor']
    ['5.6', '2.7', '4.2', '1.3', 'Iris-versicolor']
    ['5.7', '3.0', '4.2', '1.2', 'Iris-versicolor']
    ['5.7', '2.9', '4.2', '1.3', 'Iris-versicolor']
    ['6.2', '2.9', '4.3', '1.3', 'Iris-versicolor']
    ['5.1', '2.5', '3.0', '1.1', 'Iris-versicolor']
    ['5.7', '2.8', '4.1', '1.3', 'Iris-versicolor']
    ['6.3', '3.3', '6.0', '2.5', 'Iris-virginica']
    ['5.8', '2.7', '5.1', '1.9', 'Iris-virginica']
    ['7.1', '3.0', '5.9', '2.1', 'Iris-virginica']
    ['6.3', '2.9', '5.6', '1.8', 'Iris-virginica']
    ['6.5', '3.0', '5.8', '2.2', 'Iris-virginica']
    ['7.6', '3.0', '6.6', '2.1', 'Iris-virginica']
    ['4.9', '2.5', '4.5', '1.7', 'Iris-virginica']
    ['7.3', '2.9', '6.3', '1.8', 'Iris-virginica']
    ['6.7', '2.5', '5.8', '1.8', 'Iris-virginica']
    ['7.2', '3.6', '6.1', '2.5', 'Iris-virginica']
    ['6.5', '3.2', '5.1', '2.0', 'Iris-virginica']
    ['6.4', '2.7', '5.3', '1.9', 'Iris-virginica']
    ['6.8', '3.0', '5.5', '2.1', 'Iris-virginica']
    ['5.7', '2.5', '5.0', '2.0', 'Iris-virginica']
    ['5.8', '2.8', '5.1', '2.4', 'Iris-virginica']
    ['6.4', '3.2', '5.3', '2.3', 'Iris-virginica']
    ['6.5', '3.0', '5.5', '1.8', 'Iris-virginica']
    ['7.7', '3.8', '6.7', '2.2', 'Iris-virginica']
    ['7.7', '2.6', '6.9', '2.3', 'Iris-virginica']
    ['6.0', '2.2', '5.0', '1.5', 'Iris-virginica']
    ['6.9', '3.2', '5.7', '2.3', 'Iris-virginica']
    ['5.6', '2.8', '4.9', '2.0', 'Iris-virginica']
    ['7.7', '2.8', '6.7', '2.0', 'Iris-virginica']
    ['6.3', '2.7', '4.9', '1.8', 'Iris-virginica']
    ['6.7', '3.3', '5.7', '2.1', 'Iris-virginica']
    ['7.2', '3.2', '6.0', '1.8', 'Iris-virginica']
    ['6.2', '2.8', '4.8', '1.8', 'Iris-virginica']
    ['6.1', '3.0', '4.9', '1.8', 'Iris-virginica']
    ['6.4', '2.8', '5.6', '2.1', 'Iris-virginica']
    ['7.2', '3.0', '5.8', '1.6', 'Iris-virginica']
    ['7.4', '2.8', '6.1', '1.9', 'Iris-virginica']
    ['7.9', '3.8', '6.4', '2.0', 'Iris-virginica']
    ['6.4', '2.8', '5.6', '2.2', 'Iris-virginica']
    ['6.3', '2.8', '5.1', '1.5', 'Iris-virginica']
    ['6.1', '2.6', '5.6', '1.4', 'Iris-virginica']
    ['7.7', '3.0', '6.1', '2.3', 'Iris-virginica']
    ['6.3', '3.4', '5.6', '2.4', 'Iris-virginica']
    ['6.4', '3.1', '5.5', '1.8', 'Iris-virginica']
    ['6.0', '3.0', '4.8', '1.8', 'Iris-virginica']
    ['6.9', '3.1', '5.4', '2.1', 'Iris-virginica']
    ['6.7', '3.1', '5.6', '2.4', 'Iris-virginica']
    ['6.9', '3.1', '5.1', '2.3', 'Iris-virginica']
    ['5.8', '2.7', '5.1', '1.9', 'Iris-virginica']
    ['6.8', '3.2', '5.9', '2.3', 'Iris-virginica']
    ['6.7', '3.3', '5.7', '2.5', 'Iris-virginica']
    ['6.7', '3.0', '5.2', '2.3', 'Iris-virginica']
    ['6.3', '2.5', '5.0', '1.9', 'Iris-virginica']
    ['6.5', '3.0', '5.2', '2.0', 'Iris-virginica']
    ['6.2', '3.4', '5.4', '2.3', 'Iris-virginica']
    ['5.9', '3.0', '5.1', '1.8', 'Iris-virginica']
    


```python
# Divide dataset randomly into development and test datasets

ratio_factor = 0.58
devSet=[]
testSet=[]

with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                
            if random.random() < ratio_factor:
                devSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
print ('Length of devSet:')
#print (devSet)
print (len(devSet))

print ('Length of testSet:')
print(len(testSet))
```

    Length of devSet:
    94
    Length of testSet:
    56
    

## b. Implementing kNN using Distance Metric

#### b. i) Euclidean Distance


```python
# Function to find Euclidean Distance

eu_distances = []

def euclidean_distance(rec1, rec2):
    eu_dist = 0;
    
    for y in range(4):
        eu_dist += pow((rec1[y] - rec2[y]), 2)
        eu_distances.append(eu_dist)
    
    eu_dist = math.sqrt(eu_dist)
    return (eu_dist)
```

#### b. i) Normalized Euclidean Distance


```python
# Function to find Normalized Euclidean Distance

max_eu_distance = 1

def normalized_euclidean_distance(rec1, rec2):
    eu_dist = 0;
    
    for y in range(4):
        eu_dist += pow((rec1[y] - rec2[y]), 2)
    
    eu_dist = math.sqrt(eu_dist)
    
    norm_eu_dist = float(eu_dist)/max_eu_distance
    
    return (norm_eu_dist)
```

#### b. iii) Cosine Similarity


```python
# Function to find Cosine Similarity

def cosine_similarity(rec1, rec2):
    inner_product = 0;
    len_d1 = 0
    len_d2 = 0
    
    for y in range(4):
        inner_product += rec1[y] * rec2[y]
        len_d1 += pow(rec1[y],2)
        len_d2 += pow(rec2[y],2)
        
        #den = len_d1 * len_d2
        den = math.sqrt(len_d1 * len_d2)
        cosim = inner_product / den
    
    return (cosim)
```

#### b. iv) Implementing kNN


```python
# Function to find neighbors using Euclidean Distance

def checkEuNeighbors(dm,check_rec, k):
    euclidean_distances = []
    euNeighbors = []
    
    for x in range(len(devSet)):
        if(dm == 'e'):
            eu_dist = euclidean_distance(devSet[x], check_rec)
        if(dm == 'n'):
            eu_dist = normalized_euclidean_distance(devSet[x], check_rec)
        if(dm == 'c'):
            eu_dist = cosine_similarity(devSet[x], check_rec)
        
        euclidean_distances.append((devSet[x], eu_dist))
        
    euclidean_distances.sort(key=operator.itemgetter(1))
    
    #print(euclidean_distances)
    
    for y in range(k):
        euNeighbors.append(euclidean_distances[y][0])
        
    return euNeighbors;
```


```python
# Function to predict the class using Euclidean Distance

def classPredictionUsingEuDist(dm, check_rec, k):    
    euNeighbors = checkEuNeighbors(dm,check_rec, k)
    result = [rec[-1] for rec in euNeighbors]
    predict = max(set(result), key=result.count)
    return predict
```


```python
# kNN Algorithm using Euclidean Distance
def calculateEukNN(dm,k):
    predictions = []
    
    for x in range(len(devSet)):
        prediction = classPredictionUsingEuDist(dm,devSet[x], k)
        predictions.append(prediction)
        
        print(repr(devSet[x]) + '=' + repr(prediction))
        
    return(predictions)
```


```python
# Predictions using Euclidean Distance

# Predictions for k = 1
euPredictions1 = []
euPredictions1 = calculateEukNN('e',1)

# Predictions for k = 3
euPredictions3 = []
euPredictions3 = calculateEukNN('e',3)

# Predictions for k = 5
euPredictions5 = []
euPredictions5 = calculateEukNN('e',5)

# Predictions for k = 7
euPredictions7 = []
euPredictions7 = calculateEukNN('e',7)
```

    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-virginica'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-virginica'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    


```python
# Calculate max euclidean distance to use in normalization

max_eu_distance = max(eu_distances)
```


```python
# Predictions using Normalized Euclidean Distance

# Predictions for k = 1
neuPredictions1 = []
neuPredictions1 = calculateEukNN('n',1)

# Predictions for k = 3
neuPredictions3 = []
neuPredictions3 = calculateEukNN('n',3)

# Predictions for k = 5
neuPredictions5 = []
neuPredictions5 = calculateEukNN('n',5)

# Predictions for k = 7
neuPredictions7 = []
neuPredictions7 = calculateEukNN('n',7)
```

    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-virginica'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-virginica'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-setosa'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-setosa'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-setosa'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-setosa'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-setosa'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-virginica'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-versicolor'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-versicolor'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-versicolor'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-virginica'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-virginica'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-virginica'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-virginica'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-virginica'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-virginica'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-virginica'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-virginica'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-virginica'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-virginica'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-virginica'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-virginica'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-virginica'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-virginica'
    


```python
# Predictions using Cosine Similarity

# Predictions for k = 1
cosPredictions1 = []
cosPredictions1 = calculateEukNN('c',1)

# Predictions for k = 3
cosPredictions3 = []
cosPredictions3 = calculateEukNN('c',3)

# Predictions for k = 5
cosPredictions5 = []
cosPredictions5 = calculateEukNN('c',5)

# Predictions for k = 7
cosPredictions7 = []
cosPredictions7 = calculateEukNN('c',7)
```

    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-setosa'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-setosa'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-setosa'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-setosa'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-setosa'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-setosa'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-setosa'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-setosa'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-setosa'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-setosa'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-setosa'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-setosa'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-setosa'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-setosa'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-setosa'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-setosa'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-setosa'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-setosa'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-setosa'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-setosa'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-setosa'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-setosa'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-setosa'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-setosa'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-setosa'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-setosa'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-setosa'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-setosa'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-setosa'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-setosa'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-setosa'
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.4, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.4, 2.9, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.7, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.0, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [5.8, 4.0, 1.2, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.7, 4.4, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.9, 1.3, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.5, 1.4, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.5, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.4, 3.4, 1.7, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.7, 1.5, 0.4, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.3, 1.7, 0.5, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.0, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.2, 3.4, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.7, 3.2, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.8, 3.1, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.1, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.9, 3.6, 1.4, 0.1, 'Iris-setosa']='Iris-virginica'
    [4.4, 3.0, 1.3, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.4, 1.5, 0.2, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.3, 0.3, 'Iris-setosa']='Iris-virginica'
    [5.0, 3.5, 1.6, 0.6, 'Iris-setosa']='Iris-virginica'
    [5.1, 3.8, 1.6, 0.2, 'Iris-setosa']='Iris-virginica'
    [4.6, 3.2, 1.4, 0.2, 'Iris-setosa']='Iris-virginica'
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor']='Iris-setosa'
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']='Iris-setosa'
    [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor']='Iris-setosa'
    [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor']='Iris-setosa'
    [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor']='Iris-setosa'
    [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor']='Iris-setosa'
    [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor']='Iris-setosa'
    [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor']='Iris-setosa'
    [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor']='Iris-setosa'
    [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.9, 5.6, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.5, 3.0, 5.8, 2.2, 'Iris-virginica']='Iris-setosa'
    [7.3, 2.9, 6.3, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 2.5, 5.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.6, 6.1, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.8, 3.0, 5.5, 2.1, 'Iris-virginica']='Iris-setosa'
    [5.7, 2.5, 5.0, 2.0, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.8, 5.1, 2.4, 'Iris-virginica']='Iris-setosa'
    [7.7, 3.8, 6.7, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.2, 5.7, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.6, 2.8, 4.9, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.3, 2.7, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.2, 6.0, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.2, 2.8, 4.8, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.1, 3.0, 4.9, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.1, 'Iris-virginica']='Iris-setosa'
    [7.2, 3.0, 5.8, 1.6, 'Iris-virginica']='Iris-setosa'
    [7.4, 2.8, 6.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [7.9, 3.8, 6.4, 2.0, 'Iris-virginica']='Iris-setosa'
    [6.4, 2.8, 5.6, 2.2, 'Iris-virginica']='Iris-setosa'
    [6.1, 2.6, 5.6, 1.4, 'Iris-virginica']='Iris-setosa'
    [6.3, 3.4, 5.6, 2.4, 'Iris-virginica']='Iris-setosa'
    [6.4, 3.1, 5.5, 1.8, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.4, 2.1, 'Iris-virginica']='Iris-setosa'
    [6.9, 3.1, 5.1, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']='Iris-setosa'
    [6.7, 3.3, 5.7, 2.5, 'Iris-virginica']='Iris-setosa'
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']='Iris-setosa'
    [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']='Iris-setosa'
    

## c. i) Calculate Accuracy using development dataset


```python
# Function to find the accuracy using predictions

def calculateAccuracy(predictions):
    count = 0
    for x in range(len(devSet)):
        if devSet[x][-1] == predictions[x]:
            count += 1
            
        accuracy = (count/float(len(devSet))) * 100.0
    return accuracy
```


```python
# Calculate accuracy for devSet using Euclidean Distance
# for k = 1,3,5,7

euAccuracy1 = calculateAccuracy(euPredictions1)
euAccuracy3 = calculateAccuracy(euPredictions3)
euAccuracy5 = calculateAccuracy(euPredictions5)
euAccuracy7 = calculateAccuracy(euPredictions7)

print(euAccuracy1)
print(euAccuracy3)
print(euAccuracy5)
print(euAccuracy7)
```

    100.0
    97.87234042553192
    97.87234042553192
    98.93617021276596
    


```python
# Calculate accuracy for devSet using Euclidean Distance
# for k = 1,3,5,7

neuAccuracy1 = calculateAccuracy(neuPredictions1)
neuAccuracy3 = calculateAccuracy(neuPredictions3)
neuAccuracy5 = calculateAccuracy(neuPredictions5)
neuAccuracy7 = calculateAccuracy(neuPredictions7)

print(neuAccuracy1)
print(neuAccuracy3)
print(neuAccuracy5)
print(neuAccuracy7)
```

    100.0
    97.87234042553192
    97.87234042553192
    98.93617021276596
    


```python
# Calculate accuracy for devSet using Cosine Similarity
# for k = 1,3,5,7

cosAccuracy1 = calculateAccuracy(cosPredictions1)
cosAccuracy3 = calculateAccuracy(cosPredictions3)
cosAccuracy5 = calculateAccuracy(cosPredictions5)
cosAccuracy7 = calculateAccuracy(cosPredictions7)

print(cosAccuracy1)
print(cosAccuracy3)
print(cosAccuracy5)
print(cosAccuracy7)
```

    0.0
    0.0
    0.0
    0.0
    


```python

```


```python

```
