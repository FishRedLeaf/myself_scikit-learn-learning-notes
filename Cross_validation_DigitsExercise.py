# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:01:00 2018

@author: Administrator

On the digits dataset, plot the cross-validation score of a SVC estimator 
with an linear kernel as a function of parameter C 
(use a logarithmic grid of points, from 1 to 10).

"""

#print(__doc__)


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()
for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores)) #计算序列标准差

# Do the plotting
import matplotlib.pyplot as plt
plt.figure(1, figsize=(4, 3)) #figsize控制图像大小 4*3点
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
print(locs)
print(labels)
plt.yticks(locs, list(map(lambda x: "%g" % x, locs))) #对locs中的每一项进行%g处理。 
#对labels中的值，例如0.2 0.4，以较短的形式打印
#用于打印浮点型数据时，会去掉多余的零，至多保留六位有效数字（不同于%e的默认保留小数点后6位）
plt.ylabel('cross_val score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()