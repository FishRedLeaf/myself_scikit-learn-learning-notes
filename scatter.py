# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:07:07 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
plt.figure()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()