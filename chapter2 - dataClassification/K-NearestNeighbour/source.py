# Implentation of K-Nearest Neighbors (K-NN) in preprocessing of data

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('social_network_ads.csv')
X = dataset.iloc[:, [2, 3]].values # age, estimated salary
y = dataset.iloc[:, 4].values # independent variable, purchased

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # class to identify the neighbours (n_neighbours = K, metric (minkowski for euclidean distances), p=1 denotes 2D cartesian space)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) # gives prediction of y value for corresponding x values from X_test ie. : predicts if corresponding user has purchased or not

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # returns incorrect predictions by comparing to actual dataset

