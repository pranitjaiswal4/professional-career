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

---
## Download Links:
<a href="https://pranitjaiswal.netlify.com/files/Jaiswal_02.ipynb" download>Jupyter Notebook (.ipynb)</a>   

<a href="https://pranitjaiswal.netlify.com/files/bezdekIris.data" download>Input Data (.data)</a>

---

## a.	Divide the dataset as development and test


```python
# imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import math
import operator 
```


```python
# Read CSV file and print data

filename = r'Data/bezdekIris.data'

with open(filename) as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        if not row:
                continue
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

dataset=[]
ratio_factor = 0.58
devSet=[]
testSet=[]

with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        
        for row in lines:
            if not row:
                continue
            dataset.append(row)
        
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
    82
    Length of testSet:
    68
    

## b. Implementing kNN using Distance Metric

#### b. i) Euclidean Distance


```python
# Function to find Euclidean Distance

eu_distances = []

def euclidean_distance(rec1, rec2):
    eu_dist = 0;
    
    for y in range(4):
        eu_dist += pow((rec1[y] - rec2[y]), 2)
        
    eu_dist = math.sqrt(eu_dist)
    eu_distances.append(eu_dist)
    
    return (eu_dist)
```

#### b. i) Normalized Euclidean Distance


```python
# Function to find Normalized Euclidean Distance

#Reference: chap2_data.pdf by Prof. Deok Gun Park
# mahalanobis(x,y) = (x-y)^T . E^-1 . (x-y)
# E is the covarience matrix

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
        if (devSet[x] == check_rec):
            continue
        
        if(dm == 'e'):
            eu_dist = euclidean_distance(devSet[x], check_rec)
            euclidean_distances.append((devSet[x], eu_dist))
            euclidean_distances.sort(key=operator.itemgetter(1), reverse=False)
        if(dm == 'n'):
            eu_dist = normalized_euclidean_distance(devSet[x], check_rec)
            euclidean_distances.append((devSet[x], eu_dist))
            euclidean_distances.sort(key=operator.itemgetter(1), reverse=False)
        if(dm == 'c'):
            eu_dist = cosine_similarity(devSet[x], check_rec)
            euclidean_distances.append((devSet[x], eu_dist))
            euclidean_distances.sort(key=operator.itemgetter(1), reverse=True)
    
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
def calculateEukNN(dm, testDataSet ,k):
    predictions = []
    
    for x in range(len(testDataSet)):
        
        prediction = classPredictionUsingEuDist(dm,testDataSet[x], k)
        predictions.append(prediction)
        
        #print(repr(devSet[x]) + '=' + repr(prediction))
        
    return(predictions)
```


```python
# Predictions using Euclidean Distance

# Predictions for k = 1
euPredictions1 = []
euPredictions1 = calculateEukNN('e', devSet ,1)

# Predictions for k = 3
euPredictions3 = []
euPredictions3 = calculateEukNN('e', devSet ,3)

# Predictions for k = 5
euPredictions5 = []
euPredictions5 = calculateEukNN('e', devSet ,5)

# Predictions for k = 7
euPredictions7 = []
euPredictions7 = calculateEukNN('e', devSet ,7)
```


```python
# Calculate max euclidean distance to use in normalization

max_eu_distance = max(eu_distances)
```


```python
# Predictions using Normalized Euclidean Distance

# Predictions for k = 1
neuPredictions1 = []
neuPredictions1 = calculateEukNN('n', devSet ,1)

# Predictions for k = 3
neuPredictions3 = []
neuPredictions3 = calculateEukNN('n', devSet ,3)

# Predictions for k = 5
neuPredictions5 = []
neuPredictions5 = calculateEukNN('n', devSet ,5)

# Predictions for k = 7
neuPredictions7 = []
neuPredictions7 = calculateEukNN('n', devSet ,7)
```


```python
# Predictions using Cosine Similarity

# Predictions for k = 1
cosPredictions1 = []
cosPredictions1 = calculateEukNN('c', devSet ,1)

# Predictions for k = 3
cosPredictions3 = []
cosPredictions3 = calculateEukNN('c', devSet ,3)

# Predictions for k = 5
cosPredictions5 = []
cosPredictions5 = calculateEukNN('c', devSet ,5)

# Predictions for k = 7
cosPredictions7 = []
cosPredictions7 = calculateEukNN('c', devSet ,7)
```


```python

```

## C. Operations on Development Dataset

#### c. i) Calculate Accuracy using development dataset


```python
# Function to find the accuracy using predictions

def calculateAccuracy(testDataSet, predictions):
    count = 0
    for x in range(len(testDataSet)):
        if testDataSet[x][-1] == predictions[x]:
            count += 1
            
    accuracy = (count/float(len(testDataSet))) * 100.0
    return accuracy
```


```python
# Calculate accuracy for devSet using Euclidean Distance
# for k = 1,3,5,7

euAccuracy1 = calculateAccuracy(devSet, euPredictions1)
euAccuracy3 = calculateAccuracy(devSet, euPredictions3)
euAccuracy5 = calculateAccuracy(devSet, euPredictions5)
euAccuracy7 = calculateAccuracy(devSet, euPredictions7)

print(euAccuracy1)
print(euAccuracy3)
print(euAccuracy5)
print(euAccuracy7)
```

    96.34146341463415
    97.5609756097561
    96.34146341463415
    97.5609756097561
    


```python
# Calculate accuracy for devSet using Euclidean Distance
# for k = 1,3,5,7

neuAccuracy1 = calculateAccuracy(devSet, neuPredictions1)
neuAccuracy3 = calculateAccuracy(devSet, neuPredictions3)
neuAccuracy5 = calculateAccuracy(devSet, neuPredictions5)
neuAccuracy7 = calculateAccuracy(devSet, neuPredictions7)

print(neuAccuracy1)
print(neuAccuracy3)
print(neuAccuracy5)
print(neuAccuracy7)
```

    96.34146341463415
    97.5609756097561
    96.34146341463415
    97.5609756097561
    


```python
# Calculate accuracy for devSet using Cosine Similarity
# for k = 1,3,5,7

cosAccuracy1 = calculateAccuracy(devSet, cosPredictions1)
cosAccuracy3 = calculateAccuracy(devSet, cosPredictions3)
cosAccuracy5 = calculateAccuracy(devSet, cosPredictions5)
cosAccuracy7 = calculateAccuracy(devSet, cosPredictions7)

print(cosAccuracy1)
print(cosAccuracy3)
print(cosAccuracy5)
print(cosAccuracy7)
```

    96.34146341463415
    95.1219512195122
    95.1219512195122
    96.34146341463415
    

#### c. ii) Find optimal hyperparameters which give maximum accuracy


```python
# Find optimal hyperparameters which give maximum accuracy

accuracies = []

accuracies.append(['Euclidean Distance',"k=1",euAccuracy1])
accuracies.append(['Euclidean Distance',"k=3",euAccuracy3])
accuracies.append(['Euclidean Distance',"k=5",euAccuracy5])
accuracies.append(['Euclidean Distance',"k=7",euAccuracy7])

accuracies.append(['Normalized Euclidean Distance',"k=1",neuAccuracy1])
accuracies.append(['Normalized Euclidean Distance',"k=3",neuAccuracy3])
accuracies.append(['Normalized Euclidean Distance',"k=5",neuAccuracy5])
accuracies.append(['Normalized Euclidean Distance',"k=7",neuAccuracy7])

accuracies.append(['Cosine Similarity',"k=1",cosAccuracy1])
accuracies.append(['Cosine Similarity',"k=3",cosAccuracy3])
accuracies.append(['Cosine Similarity',"k=5",cosAccuracy5])
accuracies.append(['Cosine Similarity',"k=7",cosAccuracy7])

accuracies.sort(key=operator.itemgetter(2), reverse=True)
max_accuracy = accuracies[0][2]
print('Optimal hyperparameters which give maximum accuracy:')
print(accuracies[0])

accuracies.sort(key=operator.itemgetter(2), reverse=False)
min_accuracy = accuracies[0][2]
```

    Optimal hyperparameters which give maximum accuracy:
    ['Euclidean Distance', 'k=3', 97.5609756097561]
    

#### c. iii) Draw bar charts for accuracy


```python
euAc = [euAccuracy1, euAccuracy3, euAccuracy5, euAccuracy7]
neuAc = [neuAccuracy1, neuAccuracy3, neuAccuracy5, neuAccuracy7]
cosAc = [cosAccuracy1, cosAccuracy3, cosAccuracy5, cosAccuracy7]
kVal = ['k=1', 'k=3', 'k=5', 'k=7']

df = pd.DataFrame({'Eculidean': euAc, 'Normalized Euclidean': neuAc, 'Cosine': cosAc}, index=kVal)

ax = df.plot.bar()
ax.set(ylim=[min_accuracy-1, max_accuracy])
```




    [(94.1219512195122, 97.5609756097561)]




![png](output_29_1.png)


## d. Calculate final accuracy for Test Dataset


```python
# Predictions using Euclidean Distance

# Predictions for k = 1
euPredictionstestSet1 = []
euPredictionstestSet1 = calculateEukNN('e', testSet ,1)

# Predictions for k = 3
euPredictionstestSet3 = []
euPredictionstestSet3 = calculateEukNN('e', testSet ,3)

# Predictions for k = 5
euPredictionstestSet5 = []
euPredictionstestSet5 = calculateEukNN('e', testSet ,5)

# Predictions for k = 7
euPredictionstestSet7 = []
euPredictionstestSet7 = calculateEukNN('e', testSet ,7)
```


```python
# Predictions using Normalized Euclidean Distance

# Predictions for k = 1
neuPredictionstestSet1 = []
neuPredictionstestSet1 = calculateEukNN('n', testSet ,1)

# Predictions for k = 3
neuPredictionstestSet3 = []
neuPredictionstestSet3 = calculateEukNN('n', testSet ,3)

# Predictions for k = 5
neuPredictionstestSet5 = []
neuPredictionstestSet5 = calculateEukNN('n', testSet ,5)

# Predictions for k = 7
neuPredictionstestSet7 = []
neuPredictionstestSet7 = calculateEukNN('n', testSet ,7)
```


```python
# Predictions using Cosine Similarity

# Predictions for k = 1
cosPredictionstestSet1 = []
cosPredictionstestSet1 = calculateEukNN('c', testSet ,1)

# Predictions for k = 3
cosPredictionstestSet3 = []
cosPredictionstestSet3 = calculateEukNN('c', testSet ,3)

# Predictions for k = 5
cosPredictionstestSet5 = []
cosPredictionstestSet5 = calculateEukNN('c', testSet ,5)

# Predictions for k = 7
cosPredictionsv7 = []
cosPredictionstestSet7 = calculateEukNN('c', testSet ,7)
```


```python
# Calculate accuracy for testSet using Euclidean Distance
# for k = 1,3,5,7

euAccuracytestSet1 = calculateAccuracy(testSet, euPredictionstestSet1)
euAccuracytestSet3 = calculateAccuracy(testSet, euPredictionstestSet3)
euAccuracytestSet5 = calculateAccuracy(testSet, euPredictionstestSet5)
euAccuracytestSet7 = calculateAccuracy(testSet, euPredictionstestSet7)

print(euAccuracytestSet1)
print(euAccuracytestSet3)
print(euAccuracytestSet5)
print(euAccuracytestSet7)
```

    94.11764705882352
    95.58823529411765
    95.58823529411765
    95.58823529411765
    


```python
# Calculate accuracy for testSet using Euclidean Distance
# for k = 1,3,5,7

neuAccuracytestSet1 = calculateAccuracy(testSet, neuPredictionstestSet1)
neuAccuracytestSet3 = calculateAccuracy(testSet, neuPredictionstestSet3)
neuAccuracytestSet5 = calculateAccuracy(testSet, neuPredictionstestSet5)
neuAccuracytestSet7 = calculateAccuracy(testSet, neuPredictionstestSet7)

print(neuAccuracytestSet1)
print(neuAccuracytestSet3)
print(neuAccuracytestSet5)
print(neuAccuracytestSet7)
```

    94.11764705882352
    95.58823529411765
    95.58823529411765
    95.58823529411765
    


```python
# Calculate accuracy for testSet using Cosine Similarity
# for k = 1,3,5,7

cosAccuracytestSet1 = calculateAccuracy(testSet, cosPredictionstestSet1)
cosAccuracytestSet3 = calculateAccuracy(testSet, cosPredictionstestSet3)
cosAccuracytestSet5 = calculateAccuracy(testSet, cosPredictionstestSet5)
cosAccuracytestSet7 = calculateAccuracy(testSet, cosPredictionstestSet7)

print(cosAccuracytestSet1)
print(cosAccuracytestSet3)
print(cosAccuracytestSet5)
print(cosAccuracytestSet7)
```

    98.52941176470588
    98.52941176470588
    97.05882352941177
    95.58823529411765
    

