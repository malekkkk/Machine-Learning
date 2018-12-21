# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:24:16 2018

@author: Oussema
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

## Generate Database 

#X, y=make_blobs(n_samples=200, n_features=2, centers=3,random_state=None)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

## Apply K-means in order to get centers
#kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
#print('Centers : ',kmeans.cluster_centers_)
#Centers=kmeans.cluster_centers_
#Labels=kmeans.labels_

# Moyenne des centres

Center=(Centers[0]+Centers[1]+Centers[2])/3





def between_scatter_matrix(Centers,center,c):
    SB=np.zeros((2,2))
    for i in range(c):
       SB+=(Centers[i,:]-center)*(Centers[i,:]-center).T 
    return SB
 
def within_scatter_matrix(Data,Centers,labels,c):
    SW1=np.zeros((2,2))
    SW2=np.zeros((2,2))
    SW3=np.zeros((2,2))
    SW4=np.zeros((2,2))
    SW5=np.zeros((2,2))
    SW6=np.zeros((2,2))
    SW7=np.zeros((2,2))
    SW8=np.zeros((2,2))
    SW9=np.zeros((2,2))
    SW10=np.zeros((2,2))
    for m in range(labels.shape[0]):
        if labels[m]==0:
            SW1 = SW1 + (Data[m,:]-Centers[0,:])*(Data[m,:]-Centers[0,:]).T
        elif labels[m]==1:
            SW2 = SW2 + (Data[m,:]-Centers[1,:])*(Data[m,:]-Centers[1,:]).T
        elif labels[m]==2:
            SW3 = SW3 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[2,:]).T
        elif labels[m]==3:
            SW4 = SW4 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[3,:]).T
        elif labels[m]==4:
            SW5 = SW5 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[4,:]).T
        elif labels[m]==5:
            SW6 = SW6 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[5,:]).T
        elif labels[m]==6:
            SW7 = SW7 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[6,:]).T
        elif labels[m]==7:
            SW8 = SW8 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[7,:]).T
        elif labels[m]==8:
            SW9 = SW9 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[8,:]).T
        elif labels[m]==9:
            SW10 = SW10 + (Data[m,:]-Centers[2,:])*(Data[m,:]-Centers[9,:]).T
    return SW1+SW2+SW3+SW4+SW5+SW6+SW7+SW8+SW9+SW10
plt.figure(figsize=(12, 12))
n_samples =200
random_state = 100
#X, y = make_blobs(n_samples=n_samples,centers=3, random_state=random_state)
kmeans1=KMeans(n_clusters=1,random_state=0).fit(X)
center=kmeans1.cluster_centers_
for c in range (2,10):
    y_pred = KMeans(n_clusters=c ,random_state=0).fit_predict(X)
    plt.subplot(4, 3, c-1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("figure")
    kmeans=KMeans(n_clusters=c,random_state=0).fit(X)
    Centers=kmeans.cluster_centers_
    labels=kmeans.labels_
    SB= between_scatter_matrix(Centers,center,c)
    SW= within_scatter_matrix(X,Centers,labels,c)

    Sep= np.trace(SB)
    Comp= np.trace(SW)
    Validity_Index= Sep/Comp

    #print(SB,SW,Sep,Comp)
    print('VIndex',Validity_Index)