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

def rotateMatrix(mat): 
      
    N = np.size(mat,0)
    # Consider all squares one by one 
    for x in range(0, int(N/2)): 
          
        # Consider elements in group    
        # of 4 in current square 
        for y in range(x, N-x-1): 
              
            # store current cell in temp variable 
            temp = mat[x][y] 
  
            # move values from right to top 
            mat[x][y] = mat[y][N-1-x] 
  
            # move values from bottom to right 
            mat[y][N-1-x] = mat[N-1-x][N-1-y] 
  
            # move values from left to bottom 
            mat[N-1-x][N-1-y] = mat[N-1-y][x] 
  
            # assign temp to left 
            mat[N-1-y][x] = temp 

def getinfomat(tag):
    tag = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
    # print(tag[1,1])
    m = np.size(tag,0)
    print(m)
    m = (m - (m%8))/8
    print(m)
    n = np.size(tag,1)
    print(n)
    n = (n - (n%8))/8
    print(n)
    mat = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            c=0
            for a in range(int((i-1)*m),int(i*m)):
                for b in range(int((j-1)*n),int(j*n)):
                    c = c + tag[a,b]
            c = c/(m*n)
            if c < 170:
                mat[i-1,j-1] = 0
            else:
                mat[i-1,j-1]= 1

    rotateMatrix(mat)
    print(mat)
    fmat = mat[2:6,2:6]
    print(fmat)

    if fmat[0,0] == 1:
        rotateMatrix(fmat)
        rotateMatrix(fmat)
    elif fmat[3,0] == 1:
        rotateMatrix(fmat)
    elif fmat[0,3] == 1:
        rotateMatrix(fmat)
        rotateMatrix(fmat)
        rotateMatrix(fmat)

    print(fmat)
    
    id = 1*fmat[1,1] + 2*fmat[1,2] + 4*fmat[2,2] + 8*fmat[2,2]

    print(id)
    while(True):
        cv2.imshow('tag',tag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


tag = cv2.imread('/home/rohan/Desktop/ENPM673/Project1/AR Project/Reference Images/ref_marker.png')

getinfomat(tag)

