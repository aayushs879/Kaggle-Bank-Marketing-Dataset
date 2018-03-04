# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:01:41 2018

@author: aayush
"""
#Importing Libraries
import numpy as np
import pandas as pd 
import seaborn as sns

#importing Whole dataset
all_data = pd.read_csv('bank.csv')

#Checking for null values and unique values in each column
print(all_data.head())
print(all_data.info())
print(all_data.nunique())

#setting out target variable
target = all_data.iloc[:,-1].values
y = (target == 'yes').astype(np.int)

#dropping the target variable
all_data = all_data.drop(['deposit'], axis = 1)

#taking out numerical data, some columns even with num data had less values, so considered them as ordinal data
X_num = all_data.iloc[:,[0,5,11]].values

#dropping the values which were taken earlier as num data
all_data = all_data.drop(['balance', 'age', 'duration' ], axis = 1)

#taking categorical or ordinal data
X_cat = all_data.iloc[:,:]

#encoding string values in data, Please note that some values were already numeric, but still considered them so as to later perform the chi2 test
from sklearn.preprocessing import LabelEncoder, OneHotEncoder#used one hot encoding as well, but ressults were quite same and model took time for training
encode = LabelEncoder()
X_cat = X_cat.apply(encode.fit_transform)

#performing chi2 test for feature selection 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
sel = SelectKBest(chi2, 12)#this number 12 is a rsult after some hit and trials
sel.fit(X_cat, y)
score = (sel.scores_)
X_cat = sel.transform(X_cat)

#Too much of ordinal data and if it were to be one hot encoded and having applied a linear model it'd take too much time for training 
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators = 300)

#merging numerical and categorical/ordinal data
X = np.concatenate((X_cat,X_num), axis = 1)

#splitting of data into training set and test set
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(  X, y, test_size=2000, random_state=42)

#fitting the model on training set
clf1.fit(X_train, y_train)

#for evaluation
from sklearn.metrics import confusion_matrix as cm
matrix =  cm(y_test,clf1.predict(X_test) )

#checking accuracy score
accuracy = (matrix[0,0] + matrix[1,1])/2000
print(accuracy)
