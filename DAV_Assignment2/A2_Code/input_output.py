# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 01:23:00 2021

@author: cwhwe
"""

import os
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment/A2_Code")

from transformers import transformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML_Model import model
from datasets import X_train, X_test, y_train, y_test
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


type_ = input("Machine Type: ")
air_temperature = input("Air Temperature [k]: ")
process_temperature = input("Process Temperature [k]: ")
rotation_speed = input("Rotation Speed [rpm]: ")
torque = input("Torque [Nm]: ")
tool_wear = input("Tool Wear [min]: ")



model = model(X_train, y_train, X_test, y_test)

input_ = transformer(type_, float(air_temperature), float(process_temperature), float(rotation_speed),
                     float(torque), float(tool_wear))

prediction1 = model.extremeGradientBoosting("predict", input_)
prediction2 = model.randomForest("predict", input_)
prediction3 = model.supportVectorClassifier("predict", input_)

Vote = Counter([prediction1, prediction2, prediction3])
voteRank = Vote.most_common()
finalPrediction = voteRank[0][0]

print("Result: {}".format(finalPrediction))