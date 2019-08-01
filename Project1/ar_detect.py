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


print(cv2.__version__)

import matplotlib.pyplot as plt

def harrisFeatures(img):
    # img = img.deepcopy()
    Rs = cv2.cornerHarris(img,2,3,k=0.04)
    th = Rs<0.0001*Rs.max()
    m = Rs>0.0001*Rs.max()
    Rs[th] = 0
    corners = np.where(m)

    return Rs,corners



cap = cv2.VideoCapture('/home/rohan/Desktop/ENPM673/Project1/AR Project/Input Sequences/Tag0.mp4',0)
print(cap.isOpened())

detector = cv2.SimpleBlobDetector_create()
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
# params.minArea = 1000
params.maxArea = 50000
# params.maxArea = 10000

# params.filterByConvexity = True
# params.minConvexity = 0.9

# params.filterByInertia = True
# params.minInertiaRatio = 0.8
params.filterByColor = True
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

while(True):
    i = 0
    ret, frame = cap.read()
    im = frame.copy()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame,(5,5),0)
    blackmin = 180     #np.array([20,50,50])
    blackmax = 255     #np.array([60,255,255])
    mask = cv2.inRange(frame,blackmin,blackmax)
    
    res = cv2.bitwise_and(frame,frame,mask = mask)
    
    ## Blob detection
    
#     detect = cv2.SimpleBlobDetector_create(params)
    reverse_mask = mask
    keypoints = detector.detect(reverse_mask)
   
    draw = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if (keypoints[i]!=[]):    
        (x,y) = keypoints[i].pt
        print(keypoints[i].size)
        # r = np.ceil(np.sqrt(((keypoints[i].size)/np.pi)))*20
        r = np.ceil(keypoints[i].size)*0.9
        print('   ',r)
        out = res[int(y-r):int(y+r),int(x-r):int(x+r)]
#     cv2.imshow('videqo',draw)
#     print(keypoints[i].pt[0])
#     print(keypoints[i].size)
#     print(keypoints.pt)
#     print(keypoints.size)
    i += 1 
    Rs, corners = harrisFeatures(out)
    print(corners[0])
    X = corners[0]
    Y = corners[1]

    for i in range(0,len(X)):
        x = X[i]
        y = Y[i]
        cv2.circle(out,(x,y),2,255,-1)

    cv2.imshow('videq1',out)
    
#     cv2.imshow('video',res)
    if cv2.waitKey(1) & 0xff==ord('q'):
        cv2.destroyAllWindows()
        break

# def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
#     for kp in keypoints:
#         x, y = kp.pt
#         cv2.circle(vis, (int(x), int(y)), 2, color)

# while(True):
#     ret, frame = cap.read()
#     print(ret)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray,(5,5),0)
#     params = cv2.SimpleBlobDetector_Params()

#     # Change thresholds
#     params.minThreshold = 10
#     params.maxThreshold = 200


#     # Filter by Area.
#     params.filterByArea = True
#     params.minArea = 1500

#     # Filter by Circularity
#     params.filterByCircularity = True
#     params.minCircularity = 0.1

#     # Filter by Convexity
#     params.filterByConvexity = True
#     params.minConvexity = 0.87

#     # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.01

#     # Create a detector with the parameters
#     detector = cv2.SimpleBlobDetector_create(params)


#     # Detect blobs.
#     keypoints = detector.detect(blur)
#     print(keypoints)
#     # Draw detected blobs as red circles.
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
#     # the size of the circle corresponds to the size of blob

#     # im_with_keypoints = cv2.drawKeypoints(blur, keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     # draw_keypoints(blur,keypoints)
#     # cv2.imshow("Keypoints", blur)
#     # cv2.waitKey(0)
    
#     # edges = cv2.Canny(blur,240,250)
#     # cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
#     # area = [cv2.contourArea(c) for c in cnts]
#     cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#     # cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
#     # t = 0
#     # for c in cnts:
#     #     perimeter = cv2.arcLength(c,True)
#     #     if perimeter > t :
#     #         t = perimeter
#     #         max = c
#     # cnts = imutils.grab_contours(np.array(cnts))
#     # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
#     # screenCnt = None
#     # print(len(contours))
#     cv2.drawContours(gray, cnts[1], -3, (0,0,0), 4)
#     # dst = cv2.cornerHarris(gray,2,3,0.04)
#     # #result is dilated for marking the corners, not important
#     # dst = cv2.dilate(dst,None)
#     # # Threshold for an optimal value, it may vary depending on the image.
#     # frame[dst>0.01*dst.max()]=[0,0,255]

#     cv2.imshow('gray',np.hstack([gray]))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # cap.release()
# # cv2.destroyAllWindows()
# import matplotlib.pyplot as plt

# cap = cv2.VideoCapture('/home/rohan/Desktop/ENPM673/Project1/AR Project/Input Sequences/Tag0.mp4',0)
# print(cap.isOpened())

# while(True):
#     ret, frame = cap.read()
#     print(ret)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray,(5,5),0)
#     edges = cv2.Canny(blur,240,250)
#     cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     # area = [cv2.contourArea(c) for c in cnts]
#     cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#     # cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
#     # t = 0
#     # for c in cnts:
#     #     perimeter = cv2.arcLength(c,True)
#     #     if perimeter > t :
#     #         t = perimeter
#     #         max = c
#     # cnts = imutils.grab_contours(np.array(cnts))
#     # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
#     # screenCnt = None
#     # print(len(contours))
#     cv2.drawContours(gray, cnts[1], -3, (0,0,0), 4)
#     # dst = cv2.cornerHarris(gray,2,3,0.04)
#     # #result is dilated for marking the corners, not important
#     # dst = cv2.dilate(dst,None)
#     # # Threshold for an optimal value, it may vary depending on the image.
#     # frame[dst>0.01*dst.max()]=[0,0,255]
#     cv2.imshow('gray',np.hstack([gray]))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # cap.release()
# # cv2.destroyAllWindows()
