# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 19:22:40 2021

@author: User
"""

#import libraries
import pandas as pd
import numpy as np

#read csv files as as df for merge purpose
sms_train_features_df = pd.read_csv("sms_train_features.csv")
sms_train_labels_df = pd.read_csv("sms_train_labels.csv")
vocabulary = open("vocabulary.txt","r")
#merge features and labes for test data
train = pd.merge(sms_train_features_df,sms_train_labels_df,how="left")
#dropping unrelevant index indicator than seperate ham and spam classes as different data frames         
train = train.drop('Unnamed: 0',1)
spam_class = train.copy()
ham_class = train.copy()
spam_class = spam_class[spam_class["class"] == 1]
ham_class = ham_class[ham_class["class"] == 0]
#convert files as array for easier calculations
sms_train_features = pd.read_csv("sms_train_features.csv",index_col = 0).values
sms_train_labels = pd.read_csv("sms_train_labels.csv",index_col = 0).values
sms_test_features = pd.read_csv("sms_test_features.csv",index_col = 0).values
sms_test_labels = pd.read_csv("sms_test_labels.csv",index_col = 0).values
spam_class = spam_class.values
ham_class = ham_class.values

#label 0 is normal message, and label 1 is spam
def get_prior_spam(labels_csv):
    non_zeros = np.count_nonzero(labels_csv)
    prior_spam = non_zeros/len(labels_csv)
    return prior_spam

#calculates the likelihoods of words in ham or spam class and stores them in an array 
def get_likelihood(class_array):
    likelihood = []
    transpose = np.transpose(class_array)
    for i in transpose:
        count = np.count_nonzero(i == 1)
        prob = count/len(i)
        likelihood.append(prob)
    return likelihood[:-1]     #there is -1 since the last row of the transposed matrix contains labels and we dont use the in likelihood calculation  

#calculates the posteriors of words in ham or spam class and stores them in an array
def calculate_posterior(test_features_row,class_array,prior):
    posterior = 1
    likelihood = get_likelihood(class_array)
    for i in range(len(test_features_row)):
        posterior = (test_features_row[i]*(likelihood[i]+(0.0000000001)) + (1-test_features_row[i])*(1-(likelihood[i]+0.0000000001)))*posterior
    posterior = abs(posterior*prior)
    return posterior
#predics the labels 
def predict_label(sms_train_labels,sms_test_features,ham_class,spam_class):
    spam_prior = get_prior_spam(sms_train_labels)
    ham_prior = 1-spam_prior
    predicted_labels=[]
    for i in sms_test_features:
        ham_posterior = calculate_posterior(i,ham_class,ham_prior)
        spam_posterior = calculate_posterior(i,spam_class,ham_prior)
        if ham_posterior>=spam_posterior:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
    return predicted_labels
#calculates the accuracy
def calculate_accuracy(predicted_labels,sms_test_labels):
    tp = 0.0 #true positive count 
    tn = 0.0 #true negative count
    fp = 0.0 #false positive count
    fn = 0.0 #false negative count
    for i in range(len(predicted_labels)):
        prediction = predicted_labels[i]
        actual = sms_test_labels[i]
        if ((prediction == 1) & (actual == 1)):
            tp+=1
        elif ((prediction == 1) & (actual == 0)):
            fp+=1
        elif ((prediction == 0) & (actual == 0)):
            tn+=1
        elif ((prediction == 0) & (actual == 1)):
            fn+=1
    accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
    print("tp:",tp)
    print("tn:",tn)
    print("fp:",fp)
    print("fn:",fn)
    return accuracy

def feature_selection(n):
    index = 0 
    accuricies = []
    sms_train_features_df = pd.read_csv("sms_train_features.csv")
    sms_train_labels_df = pd.read_csv("sms_train_labels.csv")
    train = pd.merge(sms_train_features_df,sms_train_labels_df,how="left")
    while index < len(train):
        sms_train_features_df = pd.read_csv("sms_train_features.csv")
        sms_train_labels_df = pd.read_csv("sms_train_labels.csv")
        train = pd.merge(sms_train_features_df,sms_train_labels_df,how="left")
        train = train.drop('Unnamed: 0',1)
        train = train.iloc[index:(index+n)]
        spam_class = train.copy()
        ham_class = train.copy()
        spam_class = spam_class[spam_class["class"] == 1]
        ham_class = ham_class[ham_class["class"] == 0]
        spam_class = spam_class.values
        ham_class = ham_class.values
        sms_train_labels = pd.read_csv("sms_train_labels.csv",index_col = 0).values
        sms_test_features = pd.read_csv("sms_test_features.csv",index_col = 0).values
        sms_test_labels = pd.read_csv("sms_test_labels.csv",index_col = 0).values
        predicted_labels = predict_label(sms_train_labels,sms_test_features,ham_class,spam_class)
        accuracy = calculate_accuracy(predicted_labels,sms_test_labels)
        accuricies.append(accuracy)
        index+=n
    return accuricies
    
 
predicted_labels = predict_label(sms_train_labels,sms_test_features,ham_class,spam_class)
accuracy = calculate_accuracy(predicted_labels,sms_test_labels)
#accuracy without selecting any feature
print("ACCURACY: %",accuracy)

