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
import math
import imutils

import matplotlib.pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

cap = cv2.VideoCapture("/home/rohan/Desktop/ENPM673/Project2/DataSet/challenge_video.mp4")
K = np.array([[  1.15422732e+03,0.00000000e+00,6.71627794e+02],
              [  0.00000000e+00,1.14818221e+03,3.86046312e+02],
              [  0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist = np.array([ -2.42565104e-01,-4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
    2.20573263e-02])
while(True):

    ret,frame = cap.read()
    
    h = frame.shape[0]
    w = frame.shape[1]   
    src = np.array([[w/2,h/2+110],[w/2+200,h/2+250],[w/2-200,h/2+250]])
    region_of_interest_vertices = [
    (0, h),
    (w / 2, h / 2),
    (w, h),
]

    cropped_image = region_of_interest(
        frame, np.array([region_of_interest_vertices], np.int32),)

    # gray = frame[:,:,1]#cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow('gray',gray)
    # undist = cv2.undistort(gray,K,dist)

    # dst = np.array([[300,300],[500,300],[500,600],[300,600]])
    # H,flag = cv2.findHomography(src,dst)
    # out= cv2.warpPerspective(undist,H,((w-200,h-200)))#gray[0:h-200,0:w]#
    # # out = cv2.equalizeHist(out)
    # blur = cv2.bilateralFilter(out,9,75,75)
    # median = cv2.medianBlur(out,5)

    
    plt.figure()
    plt.imshow(cropped_image)
    # Convert to grayscale here.
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    plt.figure()
    plt.imshow(cannyed_image)
    # plt.show()

    lines = cv2.HoughLinesP(
    cannyed_image,
    rho=1,
    theta=np.pi / 180,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25)
    # print(lines)

    line_image = draw_lines(frame, lines) # <---- Add this call.
    plt.figure()
    plt.imshow(line_image)
    plt.show()

    if cv2.waitKey(1)& 0xff==ord('q'):
        cv2.destroyAllWindows()
        break
