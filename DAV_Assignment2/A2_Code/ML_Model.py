# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:15:03 2021

@author: cwhwe
"""
import os
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment/A2_Code")
from transformers import transformer
from datasets import X_train, X_test, y_train, y_test
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

#Counter(y_train.reshape([-1])), Counter(y_test.reshape([-1]))

class model:
    
    
    def __init__(self, X_train, y_train, X_test, y_test):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.map = {1: "machine failure", 0: "machine not failure"}
        
    def print_information():
        print("\n")
        print("==="*25)
        print(" The machine Leanring Models for the classification problems includes: ")
        print("==="*25)
        print("\t 1) decisionTree")
        print("\t 2) randomForest")
        print("\t 3) extremeGradientBoosting")
        print("\t 4) logisticRegression")
        print("\t 5) supportVectorClassifier")
        print("\t 6) kNeighborsClassifier")
        print("\t 7) naiveBayes")
        print("==="*25)
        print("\n")
        print("==="*25)
        print(" The Queries include: ")
        print("==="*25)
        print("\t 1) y_pred")
        print("\t 2) confusion matrix")
        print("\t 3) classification report")
        print("\t 4) area under curve")
        print("\t 5) parameter")
        print("\t 6) predict")
        print("==="*25)
        
    def decisionTree(self, query, predict = None):
        model = DecisionTreeClassifier(min_samples_leaf = 3, max_features = 6, max_depth = None, criterion = 'entropy').fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)        
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
            
    def randomForest(self, query, predict = None):
        model = RandomForestClassifier(n_estimators = 100, max_features = 3, max_depth = 100, bootstrap = True).fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
            
    def extremeGradientBoosting(self, query, predict = None):
        model = XGBClassifier(objective = "binary:logistic", n_estimators = 100, seed = 123, verbosity = 0).fit(X_train, y_train)
        y_pred = model.predict(self.X_test)
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
            
    def logisticRegression(self, query, predict = None):
        model = LogisticRegression(C=0.05179474679231213, penalty= "l2").fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])] 
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
            
    def supportVectorClassifier(self, query, predict = None):
        model = SVC(C= 1000, gamma= 0.001, kernel= 'rbf').fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
        
        
    def kNeighborsClassifier(self, query, predict = None):
        model = KNeighborsClassifier(n_neighbors=4).fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, self.y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")
            
    def naiveBayes(self, query, predict = None):
        model = GaussianNB(var_smoothing=1.0).fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        if query == "y_pred":
            return y_pred
        elif query == "confusion matrix":
            return confusion_matrix(self.y_test, y_pred)
        elif query == "classification report":
            return classification_report(self.y_test, y_pred)
        elif query == "area under curve":
            fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
            return auc(fpr, tpr)
        elif query == "parameter":
            return model.get_params()
        elif (query == "predict") & (predict is not None):
            prediction = model.predict(predict.reshape([1, -1]))
            return self.map[int(prediction[0])]
        else:
            raise ValueError("Wrong input, query must be [y_pred, confusion_matrix, area under curve, parameter, predict]")