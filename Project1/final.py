"""
ENPM673 Spring 2019: Perception for A Tagutonomous Robots
Project 1: Feducial Markers: AR Tag


Author(s):
Abhinav Modi (abhi1625@umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park

Kamakshi Jain (kamakshi@terpmail.umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park

Rohan Singh (rohan42@terpmail.umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import numpy as np
import argparse
import os, sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass


import cv2
import math
import copy
import imutils
from imutils.video import VideoStream
import argparse


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

def getinfomat(tag,point):
    # tag = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
    # print(tag[1,1])
    m = np.size(tag,0)
    m = (m - (m%8))/8
    n = np.size(tag,1)
    n = (n - (n%8))/8
    mat = np.zeros((8,8))
    # av = np.zeros([8,8])
    # tav = np.zeros([4,4])
    # cav = np.zeros([4,3])
    for i in range(8):
        for j in range(8):
            c=0
            for a in range(int((i-1)*m),int(i*m)):
                for b in range(int((j-1)*n),int(j*n)):
                    if tag[a,b] > 220:
                        c = c + tag[a,b] + 0
                        # print("white")
                    else:
                        c = c + tag[a,b]
                        # print("black")
            # print(c)
            c = c/(m*n)
            # av[i-1,j-1] = c
            if c < 170:
                mat[i-1,j-1] = 0
            else:
                mat[i-1,j-1]= 1

    fmat = mat[2:6,2:6]
    # tav = av[2:6,2:6]
    # cav[0] = np.array([0,0,tav[0,0]])
    # cav[1] = np.array([0,3,tav[0,3]])
    # cav[2] = np.array([3,3,tav[3,3]])
    # cav[3] = np.array([3,0,tav[3,0]])
    # for i in range(4):
    #     for j in range(4):
    #         if tav[i,j] > maxc:
    #             cav

    # print(point)
    p1w = copy.deepcopy(point)
    # print(fmat)
    if fmat[0,0] == 1:
        rotateMatrix(fmat)
        rotateMatrix(fmat)
        p1w[0] = point[2]
        p1w[1] = point[3]
        p1w[2] = point[0]
        p1w[3] = point[1]
    elif fmat[3,0] == 1:
        rotateMatrix(fmat)
        p1w[0] = point[3]
        p1w[1] = point[0]
        p1w[2] = point[1]
        p1w[3] = point[2]
    elif fmat[0,3] == 1:
        rotateMatrix(fmat)
        rotateMatrix(fmat)
        rotateMatrix(fmat)
        p1w[0] = point[1]
        p1w[1] = point[2]
        p1w[2] = point[3]
        p1w[3] = point[0]

    id = 1*fmat[1,1] + 2*fmat[1,2] + 4*fmat[2,2] + 8*fmat[2,1]
    # print(fmat[2,2])
    # print(p1w)

    return fmat,id,p1w

#Find Homogrpahy from the given points (p1-> target)(p2-> source)
def Estimated_Homography(p1,p2):
    A  = -np.array([
          [ -p1[0][0] , -p1[0][1] , -1 , 0  , 0 , 0 , (p2[0][0]*p1[0][0]) , (p2[0][0]*p1[0][1]),(p2[0][0]) ],
          [ 0 , 0 , 0 , -p1[0][0]  , -p1[0][1] , -1 , p2[0][1]*p1[0][0] , p2[0][1]*p1[0][1] ,p2[0][1]],
          [ -p1[1][0] , -p1[1][1] , -1 , 0  , 0 , 0 , (p2[1][0]*p1[1][0]) , (p2[1][0]*p1[1][1]),(p2[1][0])],
          [ 0 , 0 , 0 , -p1[1][0]  , -p1[1][1] , -1 , p2[1][1]*p1[1][0] , p2[1][1]*p1[1][1] ,p2[1][1]],
          [ -p1[2][0] , -p1[2][1] , -1 , 0  , 0 , 0 , (p2[2][0]*p1[2][0]) , (p2[2][0]*p1[2][1]),(p2[2][0])],
          [ 0 , 0 , 0 , -p1[2][0]  , -p1[2][1] , -1 , p2[2][1]*p1[2][0] , p2[2][1]*p1[2][1] ,p2[2][1]],
          [ -p1[3][0] , -p1[3][1] , -1 , 0  , 0 , 0 , (p2[3][0]*p1[3][0]) , (p2[3][0]*p1[3][1]),(p2[3][0])],
          [ 0 , 0 , 0 , -p1[3][0]  , -p1[3][1] , -1 , p2[3][1]*p1[3][0] , p2[3][1]*p1[3][1] ,p2[3][1]],
          ], dtype=np.float64)
    U,S,V = np.linalg.svd(A)
    X = V[:][8]/V[8][8]
    Hinv = np.reshape(X,(3,3))
    H = np.linalg.inv(Hinv)
    H = H/H[2][2]
    return H

def mat_params(K,H_cube):
    Kinv = np.linalg.inv(K)
    B_hat = np.matmul(Kinv,H_cube)
    lam = ((np.linalg.norm(np.matmul(Kinv,H_cube[:,0]))+(np.linalg.norm(np.matmul(Kinv,H_cube[:,1]))))/2)
    # print("lam", 1/lam)
    sgn = np.linalg.det(B_hat)
    if sgn<0:
        B = B_hat*-1/lam
    elif sgn>0:
        B = B_hat/lam
    r1 = B[:,0]/lam
    r2 = B[:,1]/lam
    r3 = np.cross(r1,r2)*lam
    t = np.array([B[:,2]/lam]).T
    R = np.array([r1,r2,r3]).T
    return np.hstack([R,t])


def draw_cube(K,R,p1w,p2l,im):
    P = np.matmul(K,np.matrix(R))
    P = P/P[2,3]
    pt1 = np.array([p2l[0][0],p2l[0][1],-512,1]).T
    pt2 = np.array([p2l[1][0],p2l[1][1],-512,1]).T
    pt3 = np.array([p2l[2][0],p2l[2][1],-512,1]).T
    pt4 = np.array([p2l[3][0],p2l[3][1],-512,1]).T

    p1 = np.matmul(P,pt1)
    p2 = np.matmul(P,pt2)
    p3 = np.matmul(P,pt3)
    p4 = np.matmul(P,pt4)

    cv2.line(im,tuple(p1w[0]),tuple(p1w[1]),(0,255,0),3)
    cv2.line(im,tuple(p1w[1]),tuple(p1w[2]),(0,255,0),3)
    cv2.line(im,tuple(p1w[2]),tuple(p1w[3]),(0,255,0),3)
    cv2.line(im,tuple(p1w[3]),tuple(p1w[0]),(0,255,0),3)

    cv2.line(im,tuple(p1w[0]),tuple([int(p1[0,0]/p1[0,2]),int(p1[0,1]/p1[0,2])]),(255,0,0),3)
    cv2.line(im,tuple(p1w[1]),tuple([int(p2[0,0]/p2[0,2]),int(p2[0,1]/p2[0,2])]),(255,0,0),3)
    cv2.line(im,tuple(p1w[2]),tuple([int(p3[0,0]/p3[0,2]),int(p3[0,1]/p3[0,2])]),(255,0,0),3)
    cv2.line(im,tuple(p1w[3]),tuple([int(p4[0,0]/p4[0,2]),int(p4[0,1]/p4[0,2])]),(255,0,0),3)
    cv2.line(im,tuple([int(p4[0,0]/p4[0,2]),int(p4[0,1]/p4[0,2])]),tuple([int(p1[0,0]/p1[0,2]),int(p1[0,1]/p1[0,2])]),(0,0,255),3)
    cv2.line(im,tuple([int(p1[0,0]/p1[0,2]),int(p1[0,1]/p1[0,2])]),tuple([int(p2[0,0]/p2[0,2]),int(p2[0,1]/p2[0,2])]),(0,0,255),3)
    cv2.line(im,tuple([int(p2[0,0]/p2[0,2]),int(p2[0,1]/p2[0,2])]),tuple([int(p3[0,0]/p3[0,2]),int(p3[0,1]/p3[0,2])]),(0,0,255),3)
    cv2.line(im,tuple([int(p3[0,0]/p3[0,2]),int(p3[0,1]/p3[0,2])]),tuple([int(p4[0,0]/p4[0,2]),int(p4[0,1]/p4[0,2])]),(0,0,255),3)
    return im




def causewarpPerspectivewasntgoodenoughforyou(img1,h,img2,rangex,rangey):
    # print(h)
    hinv = np.linalg.inv(h)
    # print(hinv)
    hinv = hinv/hinv[2,2]
    # print(hinv)
    for i in range(rangex[0],rangex[1]):            # x
        for j in range(rangey[0],rangey[1]):       # y

            p = np.array([i,j,1]).T
            pn = np.matmul(hinv,p)
            pn = ((pn/pn[2])).astype(int)
            # print(pn)
            if (pn[0] < img2.shape[1]) and (pn[1] < img2.shape[0]) and (pn[0] > -1) and (pn[1] > -1) and (j < img1.shape[0]) and (i < img1.shape[1]):
                img1[j,i] = img2[pn[1],pn[0]]

    return img1
############################################################################################################################################

def main():

    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--video', default='./data/Input Sequences/', help='Define name of the video you want to use')

    Args = Parser.parse_args()
    video = Args.video
    path = './data/Input Sequences/'+str(video)+'.mp4'
    #Tag Detection
    cap = cv2.VideoCapture(path)
    lena=cv2.imread("./data/AR Project/Reference Images/Lena.png")


    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output = str(video)+'.avi'
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('D','I','V','X'), 10, (frame_width,frame_height))

    #Setup Blob detector
    detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.maxArea = 50000
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)

    count = 0
    flag = True
    while(flag):
        ret, image = cap.read()
        if ret == 0:
            flag = False
            break
        im = copy.deepcopy(image)
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blackmin = 180     #np.array([20,50,50])
        blackmax = 255     #np.array([60,255,255])
        mask = cv2.inRange(frame,blackmin,blackmax)

        res = cv2.bitwise_and(frame,frame,mask = mask)


        ## Blob detection
        reverse_mask = mask
        keypoints = detector.detect(reverse_mask)

        draw = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print(len(keypoints))
        for i in range(0,len(keypoints)):
            if (keypoints[i]!=[]):
                (x,y) = keypoints[i].pt
                r = np.ceil(keypoints[i].size)*1
                #Crop the video to view only the Tag and white background
                out = res[int(y-r):int(y+r),int(x-r):int(x+r)]

            gray = out
            # cv2.imshow('videqo',draw)

            cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:]

            for c in cnts:
            # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break

            a = np.array(screenCnt)
            p1 = np.zeros([4,2])
            p1[0:4,0] = a[0:4,0,0]
            p1[0:4,1] = a[0:4,0,1]

            R = np.ones([4,2])*r
            C = np.matmul(np.ones([4,2]),([[x,0],[0,y]]))
            p1w = (np.ceil(C - R + p1)).astype(int)

            s = 100#10*(max(p1[0])-min(p1[0])).astype(int)
            p2 = np.array([[0,0],[s,0],[s,s],[0,s]])

            rangex = np.array([0,s]).astype(int)
            rangey = np.array([0,s]).astype(int)


            h = Estimated_Homography(p2,p1w)
            tr = causewarpPerspectivewasntgoodenoughforyou(np.zeros([200,200]),h,res,rangex,rangey)

            tr = tr[0:s,0:s]
            # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # tr = cv2.filter2D(tr, -1, kernel)

            if count == 10:
                cv2.imwrite('video_'+str(video)+'-Tag'+str(i)+'.png',tr)
            
            tag,id,p1w = getinfomat(tr,p1w)
            print(tag, id)

            for i in range(0,4):
                cv2.circle(im,(p1w[i,0],p1w[i,1]),3,(0,0,0),-1)

            p2l = np.array([[0,lena.shape[0]],[lena.shape[1],lena.shape[0]],[lena.shape[1],0],[0,0]])
            h2 = Estimated_Homography(p1w,p2l)

            rangex1 = np.array([min(p1w[:,0]+1), max(p1w[:,0])-1]).astype(int)
            rangey1 = np.array([min(p1w[:,1]+1), max(p1w[:,1])-1]).astype(int)

            im = causewarpPerspectivewasntgoodenoughforyou(im,h2,lena,rangex1,rangey1)


            # #Calibaration Matrix
            # K =np.array([[1406.08415449821,0,0],
            #             [2.20679787308599, 1417.99930662800,0],
            #             [1014.13643417416, 566.347754321696,1]]).T

            # H_cube = Estimated_Homography(p1w,p2l)
            # R = mat_params(K,H_cube)
            # # print("P mat",np.matmul(K,R))
            # im = draw_cube(K,R,p1w,p2l,im)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Tag ID = '+str(np.int32(id))
            oX = (x+r-25).astype(int)
            oY = (y+r-25).astype(int)
            cv2.putText(im,text,(oX,oY), font, 0.8,(128,0,128),1,cv2.LINE_AA)


        cv2.imshow('video',im)

        if count == 20:
            cv2.imwrite(str(video)+'.png',im)
        
        count = count + 1

        if cv2.waitKey(1) & 0xff==ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()
    out.release()
if __name__ == '__main__':
    main()