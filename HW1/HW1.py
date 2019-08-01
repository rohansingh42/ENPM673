import argparse
import os, sys
import pickle
import numpy as np

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import matplotlib.pyplot as plt

with open('HW1_data/data1.pkl', 'rb') as f:
     data1 = pickle.load(f)
#data1 = pickle.load('HW1_data/data1.pkl')
# print(len(data1)) 
# x = [i[0] for i in data1]
# y = [i[1] for i in data1]
# print(x.shape)
# for i in data1:
#     x, y = i
# print(x)
#X = np.stack((x,y),axis=0)
X = np.asarray(data1)
cov_mat1 = np.cov(X.T)
print(cov_mat1)
mean1 = np.mean(X,axis=0)
print(mean1)

plt.scatter(X[:,0],X[:,1])
#plt.show()
eig_vals, eig_vecs = np.linalg.eig(cov_mat1)
print(eig_vals)
print(eig_vecs)
origin = mean1
x = eig_vecs[0,:] 
y = eig_vecs[1,:] 
print(x,y)
plt.quiver(*origin, x, y, color=['r','b'], scale=2)

plt.show()
