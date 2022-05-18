# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:29:58 2021

@author: cwhwe
"""

def threshold(array, n):
    if (n<0.0) | (n>1.0):
        raise ValueError("The value should be in between 0 and 1")
    arry = array.copy()
    for i,j in enumerate(arry):
        if j >= n:
            arry[i] = 1
        else:
            arry[i] = 0
    return arry