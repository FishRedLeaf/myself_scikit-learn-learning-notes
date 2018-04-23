# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:33:13 2018

@author: Administrator
"""

from sklearn import linear_model
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
n_samples = len(iris_y)
iris_X_train = iris_X[:int(.9 * n_samples)]
iris_y_train = iris_y[:int(.9 * n_samples)]
iris_X_test = iris_X[int(.9 * n_samples):]
iris_y_test = iris_y[int(.9 * n_samples):]

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)
'''
LogisticRegression(C=100000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
'''
print(logistic.predict(iris_X_test))
print(iris_y_test)
print(logistic.score(iris_X_test,iris_y_test))