# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the dataset, remove unnecessary columns, and convert categorical data into numerical values.
2. Separate the dataset into input features (X) and target values (y), initialize theta values, and define the sigmoid function.
3. Train the Logistic Regression model using Gradient Descent by calculating predictions, computing gradients, and updating theta values iteratively.
4. Predict the output using the trained model, calculate the accuracy, and test the model with new student data samples.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DEVA ABISHEK P
RegisterNumber:  212223110008
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Placement_Data (1).csv')
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)
cat_cols = ["gender", "ssc_b", "hsc_b", "degree_t", "workex", "specialisation", "status", "hsc_s"]
for col in cat_cols:
    data[col] = data[col].astype('category')
for col in cat_cols:
    data[col] = data[col].cat.codes
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.random.randn(x.shape[1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy)

xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew1 = predict(theta, xnew1)
print("Prediction for Student 1:", y_prednew1)

xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew2 = predict(theta, xnew2)
print("Prediction for Student 2:", y_prednew2)
```

## Output:
<img width="336" height="82" alt="image" src="https://github.com/user-attachments/assets/1ee3b60d-f899-41d1-a068-75317545b842" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

