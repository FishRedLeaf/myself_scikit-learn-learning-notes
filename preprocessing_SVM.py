# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:38:19 2018

@author: Administrator
"""

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

'''
import numpy as np
a = np.array([[10,2.7,3.6],
             [-100,5,-2],
             [120,20,40]],dtype=np.float64)
print(a)
print(preprocessing.scale(a))
'''

X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,\
                          random_state=22,n_clusters_per_class=1,scale=100)
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()
#X = preprocessing.minmax_scale(X,feature_range=(-1,1))
X = preprocessing.scale(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
clf = SVC()#clf=classify
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))