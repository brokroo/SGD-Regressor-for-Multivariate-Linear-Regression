# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1. Load California housing data, select features and targets, and split into training and testing sets.
### 2. Scale both X (features) and Y (targets) using StandardScaler.
### 3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
### 4. Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sanjith.R
RegisterNumber:  212223230191

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn. metrics import mean_squared_error,accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
dataset=fetch_california_housing()
dataset
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()
df.isnull().sum()
df.shape
df.info()
df
X=df.drop(columns=['HousingPrice','AveOccup'])
X.info()
Y=df[['HousingPrice','AveOccup']]
Y.info()
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train) # training data should be fitted then transformed
X_test=scaler_X.transform(X_test) # testing data should be just transformed
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_test)
print(Y_test)
model=SGDRegressor(max_iter=1000,tol=1e-3)
multi_op_sgd=MultiOutputRegressor(model)
multi_op_sgd.fit(X_train,Y_train)
y_pred=multi_op_sgd.predict(X_test)
y_pred=scaler_Y.inverse_transform(y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,y_pred)
print(mse)
print("\nPredictions:\n",y_pred[:5])

*/
```

## Output:
![image](https://github.com/user-attachments/assets/55f26312-5825-4959-bcd3-106fba5f66c1)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
