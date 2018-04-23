# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:49:25 2018

@author: Administrator
"""

from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets
from sklearn.svm import SVC

X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))
    
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = SVC(kernel='linear')
scores = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) 
            for train, test in k_fold.split(X_digits)]
print(scores)

scores1 = cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')
print(scores1)