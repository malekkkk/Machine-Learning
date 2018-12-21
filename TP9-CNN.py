# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:33:15 2018

@author: Oussema
"""

from keras.models import Sequential
from keras.layers import Convolution2D as C2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Opti
import cv2, os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



def prepare_dataset(directory):
    paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    images = []
    labels = []
    row = 140
    col = 140
    for image_path in paths:
        image_pil = Image.open(image_path)
        image = np.array(image_pil, 'float')
        nbr = int(os.path.split(image_path)[1].split('.')[0].replace("subject",""))-1
        labels.append(nbr)
        images.append(image)
    return image_path,images,labels, row, col

directory = 'YALE/yalefaces/yalefaces'
image_path,images, labels, row, col = prepare_dataset(directory)
images = np.array(images, dtype="float") / 255.0
a=[]
for image in images:
    image = cv2.resize(image, (128,128))
    a.append(image)
images = np.array(a, dtype="float")


(trainX, testX, trainY, testY) = train_test_split(images,
	labels, test_size=0.25, random_state=42)


trainY=to_categorical(trainY,num_classes=15)
testY=to_categorical(testY,num_classes=15)
trainX=trainX.reshape((trainX.shape[0],trainX.shape[1],trainX.shape[2],1))
testX=testX.reshape((testX.shape[0],testX.shape[1],testX.shape[2],1))

model=Sequential()

model.add(C2D(32,(3,3),activation='relu',input_shape=(128,128,1)))
model.add(MaxPooling2D((2,2)))
model.add(C2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(C2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(C2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(15,activation='softmax'))
optimizer = Opti(lr=0.001)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, verbose=2, batch_size=5, epochs=200)

"""
model.save("model")
model = load_model("model")
"""
results = model.evaluate(testX, testY)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
N = 200
plt.plot(np.arange(0, N), model.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), model.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("fig.png")