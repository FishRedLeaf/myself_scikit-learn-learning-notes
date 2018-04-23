# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:35:35 2018

@author: Administrator
填充坐标轴颜色
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,2],[3,4]])
y = np.array([[11,22],[33,44]])
xx, yy = np.meshgrid(x,y)
z = xx
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx, yy, z)
plt.show()