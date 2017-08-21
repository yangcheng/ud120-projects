#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn import svm

linear_svc = svm.SVC(C=10000.0, kernel='rbf')
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
linear_svc.fit(features_train,labels_train)

pred = linear_svc.predict(features_test)
#########################################################
### your code goes here ###
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print accuracy

from operator import itemgetter
print itemgetter(10,26,50)(pred)
chris = [name for name in pred if name == 1]
print len(chris)
#########################################################


