# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:03:20 2021

@author: cwhwe
"""
import os 
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment/A2_Code")

import statsmodels.api as sm
from statsmodels.formula.api import ols
from datasets import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X_train = sm.add_constant(X_train)

model = sm.GLM(y_train, X_train, sm.families.family.Binomial()).fit()
print(model.summary())

beta = model.params

y_proba_ = model.predict(sm.add_constant(X_test))


y_pred = sm.add_constant(X_test) @ beta

y_proba = 1/ (1+ np.exp(-y_pred)) # Sigmoid Function 

print(np.sum(y_proba == y_proba_))

y_pred_bin = []

for i in y_proba:
    if i > 0.5:
        y_pred_bin.append(1)
    else:
        y_pred_bin.append(0)
        
y_pred_bin = np.array(y_pred_bin)

print(confusion_matrix(y_test, y_pred_bin))
print(classification_report(y_test, y_pred_bin))


        