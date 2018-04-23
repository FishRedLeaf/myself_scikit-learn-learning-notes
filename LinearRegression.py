# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:14:05 2018

@author: Administrator
"""

from sklearn import datasets



from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size = 0.3)
model = LinearRegression()
model.fit(data_X,data_y)
#print(model.coef_,model.intercept_) #<w,x>+b  coef_获得w，intercept_获得b
#print(model.predict(X_test))
#print(y_test)
#print(model.get_params())#返回LinearRegression()中的参数；若没有，则返回默认参数
#print(model.score(X_test,y_test))   #使用R^2 coefficient of determination判断预测值与标签的吻合程度




'''
import matplotlib.pyplot as plt
X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(X,y)
plt.show()
'''