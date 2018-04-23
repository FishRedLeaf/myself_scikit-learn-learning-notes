# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:26:34 2018

@author: Administrator
"""
import numpy as np
from sklearn import linear_model
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print('线性回归系数：\n',regr.coef_) #注意数据集有十个特征
test_predict = regr.predict(diabetes_X_test)
print('给定值：\n',diabetes_y_test)
print('预测值：\n',test_predict)
# The mean square error
meanSquareError = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
print('均方误差:\n',meanSquareError)
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
sco = regr.score(diabetes_X_test, diabetes_y_test) 
print('得分：\n',sco)


X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
import matplotlib.pyplot as plt 
#线性回归
plt.figure() 
np.random.seed(0)
for _ in range(6): 
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test)) 
    plt.scatter(this_X, y, s=3)
    
regrRidge = linear_model.Ridge(alpha=.1)
#岭回归
plt.figure() 
np.random.seed(0)
for _ in range(6): 
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regrRidge.fit(this_X, y)
    plt.plot(test, regrRidge.predict(test)) 
    plt.scatter(this_X, y, s=3) 
    
alphas = np.logspace(-4, -1, 6)
print('岭回归模型不同alpha下的得分：')
scores = [regrRidge.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,
       ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
print(scores)
best_alpha = alphas[scores.index(max(scores))]
regrRidge.alpha = best_alpha
regrRidge.fit(diabetes_X_train, diabetes_y_train)
print('Ridge的最佳系数：\n',regrRidge.coef_)

regrLasso = linear_model.Lasso()
scores = [regrLasso.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train
             ).score(diabetes_X_test, diabetes_y_test)for alpha in alphas]
print('Lasso模型不同alpha下的得分：')
print(scores)
best_alpha = alphas[scores.index(max(scores))]
regrLasso.alpha = best_alpha
regrLasso.fit(diabetes_X_train, diabetes_y_train)
'''
Lasso(alpha=0.025118864315095794, copy_X=True, fit_intercept=True,
   max_iter=1000, normalize=False, positive=False, precompute=False,
   random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
'''
print('Lasso的最佳系数：\n',regrLasso.coef_)