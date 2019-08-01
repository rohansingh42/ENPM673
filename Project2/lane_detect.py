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
import imutils

import matplotlib.pyplot as plt

cap = cv2.VideoCapture("/home/rohan/Desktop/ENPM673/Project2/DataSet/project_video.mp4")
K = np.array([[  1.15422732e+03,0.00000000e+00,6.71627794e+02],
              [  0.00000000e+00,1.14818221e+03,3.86046312e+02],
              [  0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist = np.array([ -2.42565104e-01,-4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
    2.20573263e-02])
while(True):

    ret,frame = cap.read()
    # plt.figure()
    # plt.imshow(frame)
    # plt.show()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    undist = cv2.undistort(gray,K,dist)
    
    h = gray.shape[0]
    w = gray.shape[1]
    
    # src = np.array([[w/2-50,h/2+110],[w/2+50,h/2+110],[w/2+200,h/2+250],[w/2-200,h/2+250]])
    src = np.array([[600,500],[770,500],[1050,680],[350,680]])
    print(gray.shape)
    print(src)
    dst = np.array([[300,300],[500,300],[500,600],[300,600]])
    H,flag = cv2.findHomography(src,dst)
    out= cv2.warpPerspective(undist,H,((w-200,h-200)))#gray[0:h-200,0:w]#
    # out = cv2.equalizeHist(out)
    blur = cv2.bilateralFilter(out,9,75,75)
    median = cv2.medianBlur(out,5)
    # median = cv2.equalizeHist(median)
    # cv2.imshow('video_out_undist',out)
    sobelx = cv2.Sobel(out,cv2.CV_64F,1,0)
    abs_x = np.absolute(sobelx)
    scaled = np.uint8(255*abs_x/np.max(abs_x))
    
    #Threshold X gradient
    th_min = 20
    th_max = 100
    binary_x = np.zeros([scaled.shape[0],scaled.shape[1]])
    binary_x[(scaled>=th_min)&(scaled<=th_max)]=1
    cv2.imshow('scaled',scaled)
    
    #Threshold Color Channel
    s_min = 170
    s_max = 255
    
#     mask1 = cv2.inRange(median,np.array([20,50,50]),np.array([30,255,255]))
    mask = cv2.inRange(median,150,255)
    seg = cv2.bitwise_and(median,median,mask=mask)
    blur = cv2.bilateralFilter(seg,9,75,75)
    canny = cv2.Canny(blur,175,200,None,3)
    sobel = cv2.Sobel(seg,cv2.CV_64F,0,1)
    # plt.imshow(sobel,cmap="binary")
    sobel = cv2.Sobel(sobel,cv2.CV_64F,1,0)
    
    cv2.imshow('sobel',sobel)
    cv2.imshow('video_out_undist',median)
    cv2.imshow('mask',mask)
    cv2.imshow('canny',canny)

    lines = cv2.HoughLines(canny,2,np.pi/180,100, None, 0, 0,0,45)
    # print(lines)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(out,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('houghlines',out)
    # plt.show()

    if cv2.waitKey(1)& 0xff==ord('q'):
        cv2.destroyAllWindows()
        break