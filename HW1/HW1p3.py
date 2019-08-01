import argparse
import os, sys
import pickle
import numpy as np
import random as rnd

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import matplotlib.pyplot as plt

def matrix_lstsqr(x, y):
    # Computes the least-squares solution to a linear matrix equation.
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)


with open('HW1_data/data2_new.pkl', 'rb') as f:
     data1 = pickle.load(f)

x = [i[0] for i in data1]
y = [i[1] for i in data1]
x = np.asarray(x)
y = np.asarray(y)

ds = len(data1)
t = 20 #np.std([x.T,y.T])
print(t)
s = 2
p = 0.99
e = 0.4
N = int(round(np.log(1-p)/np.log(1-(1-e)**s)))
print(N)

# inliers = np.ones(200)

for i in range(N) :
    pts = rnd.sample(range(0,ds-1),s)
    # print(pts,x[pts])
    slope, intercept = matrix_lstsqr(x[pts], y[pts])
    line_x = [round(min(x)) - 5, round(max(x)) + 5]
    line_y = [slope*x_i + intercept for x_i in line_x]
    # print(slope)
    a = -slope/(slope**2 + 1)**0.5
    b = 1/(slope**2 + 1)**0.5
    # print(a,b)
    er = np.absolute(a*x + b*y - intercept)
    inliers = [er < t]
    outliers = [er >= t]
    # plt.scatter(x[inliers],y[inliers],c='red')
    # plt.scatter(x[outliers],y[outliers],c='blue')
    # plt.plot(line_x, line_y, color='red')
    # plt.show()
    # for k in range(ds):
    #     er = np.absolute(a*x[k] + b*y[k] - intercept)
    #     inliers[k] = (er < t)
    # print(inliers)
    print(len(x[outliers]))
    if (len(x[outliers])/ds) < e :
        slope, intercept = matrix_lstsqr(x[inliers], y[inliers])
        line_x = [round(min(x)) - 5, round(max(x)) + 5]
        line_y = [slope*x_i + intercept for x_i in line_x]
        plt.scatter(x[inliers],y[inliers],c='red')
        plt.scatter(x[outliers],y[outliers],c='blue')
        plt.plot(line_x, line_y, color='red')
        break

plt.show()





