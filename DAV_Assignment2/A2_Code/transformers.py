# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:23:50 2021

@author: cwhwe
"""


import os
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment/A2_Code")

import numpy as np
import pandas as pd


def transformer(machine_type, air_temperature, process_temperature, 
                rotational_speed, torque, tool_wear):
    if machine_type == "H":
        x1 = [1, 0, 0]
    elif machine_type == "M":
        x1 = [0, 0, 1]
    elif machine_type == "L":
        x1 = [0, 1, 0]
    else:
        raise ValueError("machine type shuold be [H, M, L]")
    
    mean_std = list(zip([air_temperature, process_temperature, rotational_speed, torque, tool_wear],
                        [300.00493, 310.00556, 1538.7761, 39.98691, 107.951],
                        [2.000259, 1.483734, 179.2841, 9.969, 63.65415]))
    x2 = []
    for i, j, k in mean_std:
        ans = (i-j) / k
        x2.append(ans)
    
    return np.array(x1 + x2)
    