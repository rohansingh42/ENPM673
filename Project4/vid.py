import os, sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import math
import numpy as np
import matplotlib 
import cv2
import pickle
import glob
 
img_array = []
for frno in range(21,281):
    img = cv2.imread('./data/car_modtry5/cm'+str(frno)+'.jpg')
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('car_vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()