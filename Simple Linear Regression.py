# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:01:48 2022

@author: HT0222
"""

#Lab 2 Simple Linear Regression

#libraries importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Read the dataset
dataset = pd.read_csv('Salary_Data_Lab.csv')

# Year of experince (independent) , Salary(dependent)
X = dataset.iloc[: , :-1].values 
Y = dataset.iloc[: ,1].values

# splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split (X,Y,test_size=0.2 , random_state=0)

# Fitting Simple linear regression to training set
from sklearn.linear_model import LinearRegression 
model = LinearRegression ()
model.fit(X_train, Y_train)

# Predicting test set results
Y_predictor=model.predict(X_test)

# Visualize training set results
plt.scatter(X_train , Y_train , color='red')
plt.plot(X_train,model.predict(X_train),color='blue')
plt.title('Salary , Experience of training set')
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.show()

# Visualize test set results
plt.scatter(X_test , Y_test , color='red')
plt.plot(X_train,model.predict(X_train),color='blue')
plt.title('Salary , Experience of test set')
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.show()


#the prediction results 
model.predict([[2.2]])
model.predict([[3.9]])
model.predict([[4]])
model.predict([[9]])
model.predict([[9.6]])
model.predict([[10.3]])
model.predict([[10.7]])
model.predict([[13.4]])
model.predict([[14]])