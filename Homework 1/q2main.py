# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:57:51 2021

@author: User
"""
#import libraries
import pandas as pd
import math
from timeit import default_timer as timer

#import csv as pandas dataframe
diabetes_test_features = pd.read_csv("diabetes_test_features.csv")             
diabetes_test_labels = pd.read_csv("diabetes_test_labels.csv")
diabetes_train_features = pd.read_csv("diabetes_train_features.csv")
diabetes_train_labels = pd.read_csv("diabetes_train_labels.csv")

#merge the features and labels into single dataframe
df_test = pd.merge(diabetes_test_features, diabetes_test_labels,how="left")   
df_train = pd.merge(diabetes_train_features, diabetes_train_labels,how="left")

#dropping unrelevant unnamed index indicator
df_test = df_test.drop('Unnamed: 0',1)          
df_train = df_train.drop('Unnamed: 0',1)
diabetes_test_labels = diabetes_test_labels.drop('Unnamed: 0',1)

#calculates the euclidian distance between two point
def get_euclidean_distance (train_sample,test_sample):                             
    dist = 0.0
    for i in range(len(train_sample)-1):
        dist += (train_sample.iloc[i]-test_sample.iloc[i])**2
    return math.sqrt(dist)

#gets the k nearest neighbors of a single row of the test data
def get_neighbors (train,test_row,k):
    distances = []
    neighbors =[]
    for row in range(len(train)):
        train_row = train.iloc[row]
        eu_dist = get_euclidean_distance(train_row,test_row)
        distances.append((train_row,eu_dist))
    distances.sort(key = lambda tup: tup[1])
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#makes prediction based on a single row of the test data
def predict_classification(train, test_row, k):
    outcome = []
    neighbors = get_neighbors(train, test_row, k)
    for row in neighbors:
        outcome.append(row[-1])
    prediction = max(outcome, key=outcome.count)
    return prediction

#calculates the accuracy of KNN classifier by using confusion matrix
def get_accuracy(train,test,k):
    tp = 0.0 #true positive count 
    tn = 0.0 #true negative count
    fp = 0.0 #false positive count
    fn = 0.0 #false negative count
    for row in range(len(test)):
        test_row = test.iloc[row]
        prediction = predict_classification(train,test_row,k)
        actual = float(diabetes_test_labels.iloc[row])
        if ((prediction == 1) & (actual == 1)):
            tp+=1
        elif ((prediction == 1) & (actual == 0)):
            fp+=1
        elif ((prediction == 0) & (actual == 0)):
            tn+=1
        elif ((prediction == 0) & (actual == 1)):
            fn+=1
    accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
    return accuracy

#eleminates the features one by one and retuns accuricies as tuples in a list 
def backward_elemination (features,df_train,df_test):
    accuricies = []
    for i in features:
        train_new = df_train.copy()
        test_new = df_test.copy()
        test_new = test_new.drop(i,1)         
        train_new = train_new.drop(i,1)
        accuracy = get_accuracy(train_new,test_new,9) 
        accuricies.append((i,accuracy))
    return accuricies
        
#accuracy with full feature set 
      
accuracy = get_accuracy(df_train,df_test,9) 
print("The accuracy of the KNN classfier with k = 9 is %"+str(accuracy))

#first round of backward elemination 

features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
             "BMI","DiabetesPedigreeFunction","Age"]
print("Accuricies when corresponding feature is eleminited in first round" )
print(backward_elemination(features,df_train,df_test))

#after we saw Insulin made the biggest difference we eleminate it manually and start the second round of backward elemination
features = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "BMI","DiabetesPedigreeFunction","Age"]
df_train = df_train.drop("Insulin",1)
df_test = df_test.drop("Insulin",1)
print("Accuricies when corresponding feature is eliminated in second round" )
print(backward_elemination(features,df_train,df_test))

#after we saw Pregnancies made the biggest difference we eleminate it manually and start the third round of backward elemination
features = ["Glucose","BloodPressure","SkinThickness",
             "BMI","DiabetesPedigreeFunction","Age"]
df_train = df_train.drop("Pregnancies",1)
df_test = df_test.drop("Pregnancies",1)
print("Accuricies when corresponding feature is eleminated in third round" )
print(backward_elemination(features,df_train,df_test))
print("There isn't any improvement made elimination has stopped after second round")

#following part of the code is written for to calculate the elapsed time with different number of features
start = timer()
accuracy = get_accuracy(df_train,df_test,9) 
print("The accuracy of the KNN classfier with k = 9 is %"+str(accuracy))
end = timer()
print("time elapsed: ",end - start)

train_new = df_train.copy()
test_new = df_test.copy()
train_new = train_new.drop("Insulin",1)
test_new = test_new.drop("Insulin",1)
start = timer()
accuracy = get_accuracy(train_new,test_new,9) 
print("The accuracy of the KNN classfier with k = 9 is %"+str(accuracy))
end = timer()
print("time elapsed: ",end - start)

train_new = train_new.drop("Pregnancies",1)
test_new = test_new.drop("Pregnancies",1)
start = timer()
accuracy = get_accuracy(train_new,test_new,9) 
print("The accuracy of the KNN classfier with k = 9 is %"+str(accuracy))
end = timer()
print("time elapsed: ",end - start)

