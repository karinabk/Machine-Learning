#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:58:20 2018

@author: karina
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from functools import reduce

# Importing the dataset
def import_dataset(dataset):
    dataset = pd.read_csv(dataset , sep=";")
    
    mapping1 = {"management":0, "technician" : 1, "entrepreneur":2,"admin.":3, 
               "blue-color":4, "housemaid":5, "retired":6, "self-employed":7, "services":8, 
               "student":9, "unemployed":10, "unknown": None}             #dealing with missing categorical data
    
    mapping2 = {"divorced": 0, "married":1, "single":2, "unknown":None}
    
    mapping3 = {'secondary' : 0,'primary' : 1, 'unknown' : None,'tertiary':2}
    
    mapping4 = {"success":0, "failure" : 1, "unknown": None, "other":None}     
    
    dataset['job'] = dataset['job'].map(mapping1)
    dataset['marital'] = dataset['marital'].map(mapping2)
    dataset['education'] = dataset['education'].map(mapping3)
    dataset['poutcome'] = dataset['poutcome'].map(mapping4)
    
    X = dataset.iloc[:, [i != 8 for i in range(16)]].values
    y = dataset.iloc[:, -1].values
    return X, y

def imputer(X):
    #fill in empty values
    imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
    imp = imp.fit(X[:, [1, 2, 3, -1]])
    X[:, [1, 2, 3, -1]] = imp.transform(X[:, [1, 2, 3, -1]])
    return X

def encoder(X, y):
    #label encoding
    label_encoder_X = LabelEncoder()
    X[:, 4] = label_encoder_X.fit_transform(X[:, 4])     
    X[:, 6] = label_encoder_X.fit_transform(X[:, 6])    
    X[:, 7] = label_encoder_X.fit_transform(X[:, 7])         
    X[:, 9] = label_encoder_X.fit_transform(X[:, 9])     
    one_hot_encoder = OneHotEncoder(categorical_features=[1, 2, 3, 9, -1])   #create an OneHotEncoder object specifying the column
    X = one_hot_encoder.fit_transform(X).toarray()              #OneHot encode
    label_encoder_y = LabelEncoder()                            #same operations for the values which we want to predict
    y = label_encoder_y.fit_transform(y)
    return X, y

def split(X, y):
    # Splitting the dataset into the Training set and Test set
    return train_test_split(X, y, test_size = 0.25)

def scale(X_train, X_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test



from sklearn.neighbors import KNeighborsClassifier 

def train(X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    
    return classifier

def conf_matrix():
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
def preprocessing(dataset):
    X, y = import_dataset(dataset)
    X = imputer(X)
    X, y = encoder(X, y)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test = scale(X_train, X_test)
    return X_train, X_test, y_train, y_test
    
if __name__ == '__main__':
    dataset = "bank-full.csv"
    accuracies = []
    
    X_train, X_test, y_train, y_test = preprocessing(dataset)
#    classifier = train(X_train, y_train)
#    
#    y_pred = classifier.predict(X_test)
#    y_pred = (y_pred>0.5)
#    accuracy = accuracy_score(y_test, y_pred)
    
    for i in range(10): 
        X_train, X_test, y_train, y_test = preprocessing(dataset)
        classifier = train(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
#        
    median_of_accuracies = reduce(lambda x, y: x + y, accuracies) / float(len(accuracies))
    
    #n_neighbors = 5 metric = minkowski p=2    0.8924
    
    
