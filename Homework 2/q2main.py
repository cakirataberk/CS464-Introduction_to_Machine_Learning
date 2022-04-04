# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:03:55 2021

@author: User
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Part 1 Linear Regression

"""

#read data as matrix form from csv 
features_df = pd.read_csv("question-2-features.csv")
features = features_df.values
labels_df = pd.read_csv("question-2-labels.csv")
labels = labels_df.values

#calculate X^TX matrix
xtx = np.dot(features.transpose(),features)
rank = np.linalg.matrix_rank(xtx)

#exract LSTAT to use only feature for training
lstat = features_df[["LSTAT"]].values
train_features = np.ones((506,2))
train_features[:,1] = lstat[:,0]

#calculate B=(X^TX)^-1(X^TY)
reverse_xtx = np.linalg.inv(np.dot(train_features.transpose(),train_features))
xty = np.dot(train_features.transpose(),labels)
beta0,beta1 = np.dot(reverse_xtx,xty)

#predicted_labels = np.dot(lstat,beta)
predicted_labels_lin = lstat*beta1+beta0

#calculate the MSE
mse= np.sum((labels - predicted_labels_lin)**2) / len(labels)
print("MSE of the linear model is "+ str(mse))

"""
Part 2 Polynomial Regression

"""
#exract LSTAT to use only feature for training
train_features = np.ones((506,3))
train_features[:,1] = lstat[:,0]
train_features[:,2] = (lstat[:,0])**2
#calculate B=(X^TX)^-1(X^TY)
reverse_xtx = np.linalg.inv(np.dot(train_features.transpose(),train_features))
xty = np.dot(train_features.transpose(),labels)
beta0,beta1,beta2 = np.dot(reverse_xtx,xty)

#predicted_labels = np.dot(lstat,beta)
predicted_labels_pol = (lstat**2)*beta2+lstat*beta1+beta0

#calculate the MSE
mse= np.sum((labels - predicted_labels_pol)**2) / len(labels)
print("MSE of the polynomial model is "+ str(mse))

#Plot ground truth labels and predicted labels
plt.scatter(lstat,labels , color = "blue")
plt.scatter( lstat,predicted_labels_pol, s=50,color = "red")
plt.plot( lstat,predicted_labels_lin, color = "orange")
plt.title("Price vs LSTAT")
plt.xlabel("LSTAT")
plt.ylabel("Price")
plt.legend(["predicted labels(linear regression)",'true labels','predicted labels(polynomial regression)'])
plt.show()





