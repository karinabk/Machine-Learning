#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:40:54 2018

@author: karina
"""
#Artificial Neural Network implementation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

#Working with categorical data
from sklearn.preprocessing import LabelEncoder
X_1=LabelEncoder()
X[:,1]=X_1.fit_transform(X[:,1])
X_2=LabelEncoder()
X[:,2]=X_2.fit_transform(X[:,2])

#dummy variables for countries
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(categorical_features =[1])
X=encoder.fit_transform(X).toarray()
X=X[:,1:]

#Creating test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#creating first input and hidden layer
layers = Sequential()
layers.add(Dense(units=6, kernel_initializer ='glorot_uniform', activation='relu', input_dim=11 ))
#creatinf second hidden layer
layers.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
#creating output layer
layers.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))
#compiling
layers.compile(optimizer ='adam', metrics =['accuracy'], loss='binary_crossentropy' )

layers.fit(X_train,y_train,batch_size=10,epochs=100)
#getting predicted probabilities
y_prediction = layers.predict(X_test)
y_prediction = (y_prediction>0.5)

#Confusion matrics
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test,y_prediction)


