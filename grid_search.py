# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:37:27 2018

@author: Administrator
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import datasets
from sklearn.svm import SVC

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = SVC(kernel = 'linear')

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs))
clf.fit(X_digits[:1000], y_digits[:1000])        
print(clf.best_score_)                                 
print(clf.best_estimator_.C)
train_scores = clf.score(X_digits[1000:], y_digits[1000:])
print(train_scores)
#By default, the GridSearchCV uses a 3-fold cross-validation.
scores = cross_val_score(clf, X_digits, y_digits)
print(scores)  