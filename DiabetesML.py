# -*- coding: utf-8 -*-
"""

@author: Talha.Iftikhar
"""

## ML Basics 

import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os as wd
from sklearn.model_selection import GridSearchCV
import numpy as np

diab_data=pd.read_csv('pima-indians-diabetes.csv') # Read in the data

diab_data.head() # See few rows 

diab_data.shape # Dimensions of data

features=diab_data.drop('HasDiabetes',axis=1) # For features train , we remove predictor

Predictor=diab_data['HasDiabetes'].values # Prediction variable

# SKLearn data split function
features_train,features_test,pred_train,pred_test=train_test_split(features,Predictor,test_size=0.3,random_state=1,stratify=Predictor)

# Initially setting for 3 neighbors
knnML=KNeighborsClassifier(n_neighbors=3)

# Fit the model
knnML.fit(features_train,pred_train)

# See few rows , 1 donates person is diabetic and vice versa.
knnML.predict(features_test)[0:10]

# Check Accuracy of column
knnML.score(features_test,pred_test) # 64.23% 

## A step to improve accuracy


knnML2 = KNeighborsClassifier() #create new a knn model

param_grid = {'n_neighbors': np.arange(1, 25)} # Dictionary of all values we want to test for n_neighbors

knnML_gscv = GridSearchCV(knnML2, param_grid, cv=5) # test all values for n_neighbors

knnML_gscv.fit(features_train,pred_train) #fit model to data

knnML_gscv.best_params_ # n=11

# training with 11 neighbors
knnML=KNeighborsClassifier(n_neighbors=11)

knnML.fit(features_train,pred_train)

knnML.predict(features_test)[0:10]

knnML.score(features_test,pred_test)# 70.56 , not bad improvement with few lines of code.