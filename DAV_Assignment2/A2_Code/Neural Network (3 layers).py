# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:46:53 2021

@author: cwhwe
"""

import os
os.chdir("C:/Users/cwhwe/Desktop/May_2021/Data_Analytics/Assignment")

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from A2_Code.function1 import threshold
from A2_Code.datasets import X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

tf.disable_eager_execution()

num_feature = X_train.shape[1]

X = tf.placeholder(tf.float32, [None, num_feature])
y = tf.placeholder(tf.float32, [None, 1])
y_cls = tf.placeholder(tf.float32, [None])

num_w1 = 50
num_w2 = 50  #num_w1 = num_w2
num_w3 = 50
num_w4 = 1

weights_1 = tf.Variable(tf.random_normal([num_feature, num_w1]))
bias_1 = tf.Variable(tf.zeros([num_w1]))
h1 = tf.nn.relu(tf.matmul(X, weights_1) + bias_1)

weights_2 = tf.Variable(tf.random_normal([num_w1, num_w2]))
bias_2 = tf.Variable(tf.zeros([num_w2]))
h2 = tf.nn.relu(tf.matmul(h1, weights_2) + bias_2)

weights_3 = tf.Variable(tf.random_normal([num_w2, num_w3]))
bias_3 = tf.Variable(tf.zeros([num_w2]))
h3 = tf.nn.relu(tf.matmul(h2, weights_3) + bias_3)

weights_4 = tf.Variable(tf.random_normal([num_w3, num_w4]))
bias_4 = tf.Variable(tf.zeros([num_w4]))
logits = tf.matmul(h3, weights_4) + bias_4
y_pred = tf.nn.sigmoid(logits)
y_pred_ = tf.reshape(y_pred, [-1,])
y_pred_cls = tf.round(y_pred)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) 
cost = tf.reduce_mean(cross_entropy) #average of all cross-entropy values
optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.1).minimize(cost) 
correct_prediction = tf.equal(tf.reshape(y_pred_cls, [-1,]), tf.reshape(y, [-1,])) #vector of boolean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 8000

for i in range(1000):
    random_ = np.random.choice(len(X_train), batch_size)
    x_batch, y_batch = X_train[random_], y_train[random_]
    feed_dict_train = {X: x_batch, y: y_batch}
    sess.run(optimizer, feed_dict = feed_dict_train)
    
    if (i%30) == 0:
        print("Epoch {}, the accuracy is {}".format(i, sess.run(accuracy, feed_dict = {X: X_test, y: y_test})))
        
y_pred = sess.run(y_pred, feed_dict = {X: X_test})

y_predd = threshold(y_pred, 0.5)

print(confusion_matrix(y_test, y_predd))

print(classification_report(y_test, y_predd))