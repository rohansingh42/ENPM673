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
    img = cv2.imread('./results/km/'+str(frno)+'.jpg')
    if img is None:
        continue
    img = cv2.resize(img,(480,320))
    cv2.imwrite('./results/det_and_class/frame' + str(frno) + '.png',img)
 