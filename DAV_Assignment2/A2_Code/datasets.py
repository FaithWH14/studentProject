# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:23:50 2021

@author: cwhwe
"""


import os
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("ai4i2020.csv", index_col = 0)
df.columns = ["product_id", "type", "air_temperature", "process_temperature",
              "rotational_speed", "torque", "tool_wear", "machine_failure",
              "TWF", "HDF", "PWF", "OSF", "RNF"]

X = df.drop(["product_id", "machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis = 1).values
y = df.loc[:,"machine_failure"].values


x1 = X[:, 0] #categorical  << Encode
x2 = X[:, 1:] #number   << Scale

x1 = x1.reshape([-1, 1])
X_bin = OneHotEncoder().fit_transform(x1).toarray()
X_scale = StandardScaler().fit_transform(x2)

X = np.concatenate([X_bin, X_scale], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

Xy = np.concatenate([X_train, y_train.reshape([-1,1])], axis = 1)
np.random.shuffle(Xy)

X_train = Xy[:, :-1]
y_train = Xy[:, -1].reshape([-1,1])

y_test = y_test.reshape([-1, 1])
#X_bin = OneHotEncoder().fit_transform(X[:, 0].reshape([-1, 1])).toarray()