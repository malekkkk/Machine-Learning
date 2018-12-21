from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


print("-----------------------------------------")
print("Compte Rendu Boujelben Oussema - Aim - INDP3")
print("-----------------------------------------")

# Question 2
im=Image.open("C:/Users/Oussema/Desktop/Compte Rendu TP/YALE/YALE/faces/subject01.centerlight")
#im.show()
im_array=np.array(im) 


# Question 3

#vectligne=im_array.flatten()
vectligne=np.reshape(im_array,im_array.shape[0]*im_array.shape[1])
#print(vectlinge)


        
        
# Question 4

import os


tab=[]
path = "C:/Users/Oussema/Desktop/Compte Rendu TP/YALE/YALE/faces/"
y = []
i=0
k=0

for filename in os.listdir(path):
    if filename.endswith(".pgm") :
        continue
    else:
        im = Image.open(path+filename)
        im_array=np.array(im)
        vect = np.reshape(im_array,im_array.shape[0]*im_array.shape[1])
        tab.append(vect)  
        y.append(k)
    if(i==10):
        k+=1
        i=-1
    i+=1

y=y+np.array(1)
y=np.asarray(y)

# Question 4 PCA
pca = PCA(n_components=60)
pca.fit(tab)
tab1= []
tab1= pca.transform(tab)

# Question 5 

X_train, X_test, y_train, y_test = train_test_split(tab1, y, test_size = 1/3)
classifier = SVC(kernel = 'linear', C=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Question 6
print("PCA method acc : ",100*accuracy_score(y_test,y_pred))

# Question 7

lda= LDA()
lda_X=lda.fit_transform(tab,y)
X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(lda_X, y, test_size = 1/3)
lda=lda.fit(X_train_L,y_train_L)
y_pred_L=lda.predict(X_test_L)
print("LDA method acc : " ,100*accuracy_score(y_test_L,y_pred_L))

print("-----------------------------------------")
print("Compte Rendu Boujelben Oussema - Aim - INDP3")
print("-----------------------------------------")

