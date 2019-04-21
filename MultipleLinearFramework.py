#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:53:24 2019

@author: sionkang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#   generate predicted values of y using theta, X, n
def prediction(theta, X):
    pred = np.dot(X, theta.T)
    
    return pred

#   by iterating through the cost function num_iter amount of times, updates theta and shows cost converging
def GradientDescent(theta, alpha, num_iters, pred, X, y):
    cost = np.zeros(num_iters)

    for i in range(num_iters):
        diff = pred - y
        theta = theta - alpha*(1/X.shape[0])*np.dot(diff.T,X)        
        pred = prediction(theta, X)
        cost[i] = (1/(2*X.shape[0]))*np.sum(np.square((pred-y)))     
    
    return theta, cost

#   sets the size of theta, and does the prediction and gradient descent functions
def Multi_Linear_Regression(X, y, alpha, num_iters):
    n = X.shape[1] # number of features [0] = row, [1] = column
    theta = np.zeros(n)
    
    pred = prediction(theta, X)
    
    theta, cost = GradientDescent(theta, alpha, num_iters, pred, X, y)
    
    return theta, cost

#   finds the rmse value using calculated predicted y values
def rmse_metric(fin_theta, X, y):
    J = 0.0
    thetapred = fin_theta
    yprediction = np.dot(X, thetapred)
    
    J = np.sum(np.square((yprediction - y)))
        
    J = J/len(y)
    
    J = np.sqrt(J)
    
    return J

#   finds the accuracy percentage of predicted y values in comparison to the actual y values
def Accuracy(fin_theta, X , y):
    accuracy = 0.0
    thetapred = fin_theta
    yprediction = np.dot(X, thetapred)
    
    accuracy = np.sum(np.abs((yprediction - y)/(y)))
    
    accuracy = (accuracy/len(y)) * 100
    
    return accuracy

#   organizes all functions as one
def evaluateData(dataset, split, num_iters, alpha):
#    data_features = ['transaction','houseage','distancetoMRT','conveniencestores','latitude','longitude','houseprice']
    data_features = ['AT','V','AP','RH','PE']
    X = data[data_features[:4]]
    y = data[data_features[-1]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    meanTrain = np.ones(X_train.shape[1])
    stdTrain = np.ones(X_train.shape[1])
    meanTest = np.ones(X_test.shape[1])
    stdTest = np.ones(X_test.shape[1])
    
    #Normalize X_train, X_test
    for i in range(X_train.shape[1]):
        meanTrain[i] = np.mean(X_train.T[i])
        stdTrain[i] = np.std(X_train.T[i])
    
    Normalized_X_train = np.ones(X_train.shape)
    for j in range(X_train.shape[1]):
        Normalized_X_train[:,j] = (X_train.T[j] - meanTrain[j])/stdTrain[j]
        
    for i in range(X_test.shape[1]):
        meanTest[i] = np.mean(X_test.T[i])
        stdTest[i] = np.std(X_test.T[i])
    
    Normalized_X_test = np.ones(X_test.shape)
    for j in range(X_test.shape[1]):
        Normalized_X_test[:,j] = (X_test.T[j] - meanTest[j])/stdTest[j]
    
    one_columnTrain = np.ones((X_train.shape[0],1))
    Normalized_X_train = np.concatenate((one_columnTrain, Normalized_X_train), axis = 1)
    one_columnTest = np.ones((X_test.shape[0],1))
    Normalized_X_test = np.concatenate((one_columnTest, Normalized_X_test), axis = 1)
        
    theta, cost = Multi_Linear_Regression(Normalized_X_train, y_train, alpha, iters)
    
    fig, ax = plt.subplots()  
    ax.plot(np.arange(iters), cost, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch')  

    Jtrain = rmse_metric(theta, Normalized_X_train, y_train)
    accuracyTrain = Accuracy(theta, Normalized_X_train, y_train)
    
    Jtest = rmse_metric(theta, Normalized_X_test, y_test)
    accuracyTest = Accuracy(theta, Normalized_X_test, y_test)
    
    return Jtrain, 100 - accuracyTrain, Jtest, 100 - accuracyTest
    
#TESTCLASS
#data = pd.read_csv('RealEstate.csv')
data_features = ['AT','V','AP','RH','PE']
data = pd.read_csv('electricityNEW.csv')


iters = 300000
alpha = 0.0001
split = 0.3

Jtrain, accuracyTrain, Jtest, accuracyTest = evaluateData(data, split, iters, alpha)

print()
print("Train Data:")
print("RMSE Value: " + str(round(np.asscalar(Jtrain),2)))
print("Accuracy: " + str(round(np.asscalar(accuracyTrain),2)) + "%")
print()
print("Test Data:")
print("RMSE Value: " + str(round(np.asscalar(Jtest),2)))
print("Accuracy: " + str(round(np.asscalar(accuracyTest),2)) + "%")




