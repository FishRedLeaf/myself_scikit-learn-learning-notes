# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:47:10 2018

@author: Administrator
"""

from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# method 1: pickle
import pickle
# save
with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
# restore
with open('clf.pickle', 'rb') as f:
   clf2 = pickle.load(f)
   print(clf2.predict(X[0:1]))

# method 1: pickle另一种使用方式
import pickle
# save
s = pickle.dumps(clf)
#restore
clf22 = pickle.loads(s)
print(clf22.predict(X[0:1]))


# method 2: joblib
from sklearn.externals import joblib
# Save
joblib.dump(clf, 'clf.pkl')
# restore
clf3 = joblib.load('clf.pkl')
print(clf3.predict(X[0:1]))