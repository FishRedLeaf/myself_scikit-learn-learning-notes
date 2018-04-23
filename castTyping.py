# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:12:29 2018

@author: Administrator
"""

import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)#设置种子，即the starting point for a sequence of pseudorandom number
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)
'''
In this example, X is float32, which is cast to float64 by fit_transform(X).
'''


from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)  
print(list(clf.predict(iris.data[:3])))

clf.fit(iris.data, iris.target_names[iris.target])  
print(list(clf.predict(iris.data[:3])))  
'''
Here, the first predict() returns an integer array, 
since iris.target (an integer array) was used in fit. 
The second predict() returns a string array, 
since iris.target_names was for fitting.
'''