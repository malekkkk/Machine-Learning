# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:26:04 2018

@author: Oussema
"""

#Neural network module
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report




iris = load_iris()
x=iris['data']
y=iris['target']

labelbinarizer = LabelBinarizer()
y = labelbinarizer.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1/3)
########    
    


#########
def baseline_model(X_train,y_train):
	# create model
    model = Sequential()
    model.add(Dense(activation="relu", input_dim=4, units=8, kernel_initializer="uniform"))
    model.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    model.add(Dense(activation="softmax", units=3, kernel_initializer="uniform"))
	# Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 10, epochs = 50)
    return model
########
########
model = baseline_model(X_train, y_train)
y_pred = model.predict(X_test)
res1=labelbinarizer.inverse_transform(y_test)
res2=labelbinarizer.inverse_transform(y_pred)
cm = confusion_matrix(res1, res2)
print(classification_report(y_true=res1,y_pred= res2))
print(accuracy_score(y_true=res1,y_pred= res2))
#########

