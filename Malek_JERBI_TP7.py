# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:32:53 2018

@author: Oussema
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
type(iris)
print(iris.data)


x=iris['data']
y=iris['target']


#X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1/3)

mlp = MLPClassifier(hidden_layer_sizes=(120,10,10),solver='sgd',activation='relu',learning_rate_init=0.01,max_iter=500)


# Train the model
mlp.fit(X_train, y_train)


Ypredict=mlp.predict(X_test)

print("Accuracy =  " ,100*accuracy_score(y_test,Ypredict))