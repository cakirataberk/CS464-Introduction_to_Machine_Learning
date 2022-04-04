# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:21:52 2021

@author: User
"""

#import libraries
import pandas as pd
import numpy as np
import random

train_features = pd.read_csv("question-3-features-train.csv").values
test_features = pd.read_csv("question-3-features-test.csv").values
train_labels = pd.read_csv("question-3-labels-train.csv").values
test_labels = pd.read_csv("question-3-labels-test.csv").values

def find_min_max(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = []
        for row in dataset:
            col_values.append(row[i])
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
#normalize dataset
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        
normalize_dataset(train_features,find_min_max(train_features))
normalize_dataset(test_features,find_min_max(test_features))

def sigmoid(z):
    sig = (1/(1+np.exp(-z)))
    return sig

def model_fit_full(features, labels, rate,iterations):
    weights = np.zeros(len(features[0])+1) 
    beta0 = weights[0]
    betai = weights[1:len(weights)]
    for iteration in range(iterations):
        z = beta0 + np.dot(features, betai)
        predicted_labels = sigmoid(z)
        error = np.subtract(labels,predicted_labels)[:,0]
        dgradient = np.dot(features.T, error)
        beta0 += rate * np.sum(error)
        betai += rate * dgradient
    predicted_weights = np.append(beta0, betai)
    return predicted_weights

def create_subsets(features,labels,batch_size):
    #creates random subsets of train dataset for mini batch gradient descent where size of subset equals to 100 
    index = random.randint(0,612)
    mini_train_features = train_features[index:index+batch_size]
    mini_train_labels = train_labels[index:index+batch_size]
    return mini_train_features,mini_train_labels

def model_fit_mini(features, labels, rate,iterations,batch_size):
    weights = np.random.normal(0, 0.1, len(features[0])+1) 
    beta0 = weights[0]
    betai = weights[1:len(weights)]
    for iteration in range(iterations):
        mini_train_features,mini_train_labels = create_subsets(features,labels,batch_size)
        z = beta0 + np.dot(mini_train_features, betai)
        predicted_labels = sigmoid(z)
        error = np.subtract(mini_train_labels,predicted_labels)[:,0]
        dgradient = np.dot(mini_train_features.T, error)
        beta0 += rate * np.sum(error)
        betai += rate * dgradient
    predicted_weights = np.append(beta0, betai)
    return predicted_weights
    
def predict(test_features,weights):
    beta0 = weights[0]
    betai = weights[1:len(weights)]
    predicted_labels = []
    for c in range(len(test_features)):
        score = beta0 + np.sum(np.dot(test_features[c], betai))
        prob_true = sigmoid(score)
        prob_false = 1-sigmoid(score)
        if prob_false >= prob_true:
            prediction = 0
        else:
            prediction = 1
        predicted_labels.append(prediction)
    return predicted_labels

def calculate_accuracy(predicted_labels,test_labels):
    tp = 0.0 #true positive count 
    tn = 0.0 #true negative count
    fp = 0.0 #false positive count
    fn = 0.0 #false negative count
    for i in range(len(predicted_labels)):
        prediction = predicted_labels[i]
        actual = test_labels[i]
        if ((prediction == 1) & (actual == 1)):
            tp+=1
        elif ((prediction == 1) & (actual == 0)):
            fp+=1
        elif ((prediction == 0) & (actual == 0)):
            tn+=1
        elif ((prediction == 0) & (actual == 1)):
            fn+=1
    accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
    return accuracy,tp,tn,fp,fn

def performance_metrics(tp,tn,fp,fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    NPV = tn/(tn+fn)
    FPR = fp/(fp+tn)
    FDR = fp/(fp+tp)
    F1_score = 2*precision*recall/(precision+recall)
    F2_score = 5*precision*recall/(4*precision+recall)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("NPV: ",NPV)
    print("FPR: ",FPR)
    print("FDR: ",FDR)
    print("F1 score: ",F1_score)
    print("F2_score: ",F2_score)


learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for rate in learning_rates:
    weights = model_fit_full(train_features, train_labels,rate,1000)
    predicted_labels = predict(test_features,weights)
    accuracy,tp,tn,fp,fn = calculate_accuracy(predicted_labels,test_labels)
    print("When learning rate is equal to "+str(rate))
    print("Accuracy: %"+str(accuracy))
    print("tp: "+str(tp))
    print("tn: "+str(tn))
    print("fp: "+str(fp))
    print("fn: "+str(fn))

print("Performance metrics for full gradient ascent when learning rate is equal to 1e-3")
weights = model_fit_full(train_features, train_labels,1e-3,1000)
predicted_labels = predict(test_features,weights)
accuracy,tp,tn,fp,fn = calculate_accuracy(predicted_labels,test_labels)
performance_metrics(tp,tn,fp,fn)
print("Performance metrics for mini batch gradient ascent when learning rate is equal to 1e-3 and batch size is equal to 100")
weights = model_fit_mini(train_features, train_labels,1e-3,1000,100)
predicted_labels = predict(test_features,weights)
accuracy,tp,tn,fp,fn = calculate_accuracy(predicted_labels,test_labels)
print("Accuracy: %"+str(accuracy))
print("tp: "+str(tp))
print("tn: "+str(tn))
print("fp: "+str(fp))
print("fn: "+str(fn))
performance_metrics(tp,tn,fp,fn)

#
#    weights = np.zeros(len(train_features[0])+1) 
#    beta0 = weights[0]
#    betai = weights[1:len(weights)]
#    z = beta0 + np.dot(features, betai)
#    predicted_labels = sigmoid(z)
#    error = np.subtract(train_labels,predicted_labels)[:,0]





