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
for frno in range(0,2861):
    img = cv2.imread('./results/det_and_class/frame'+str(frno)+'.png')
    if img is None:
        continue
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('output_vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()