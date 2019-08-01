import sys,os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('/opt/ros/kinetic/lib/python3.3/dist-packages')

except:
    pass

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

def getCameraMatrix(path):
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
    K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
    return K, LUT


def undistortImageToGray(img,LUT):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray1)
    return gray

def features(img1, img2, K):
    #Find keypoints and feature descriptors with sift
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    orb = cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)

    #Using FLANN - Fast Library for Approximate Nearest Neighbours

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    flag = True
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # matches = sorted(matches,key = lambda x : x.distance)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(matches[0])
    inliers = []
    pts1 = []
    pts2 = []
    # print "Matches",(len(matches))
    if len(matches)==0:
        flag = False
        return flag,pts1,pts2
    # print len(kp1)
    #ratio test for selecting matches lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            inliers.append(m)
            if m.trainIdx >= len(kp1):
                continue
            elif m.trainIdx >= len(kp2):
                continue
            # print "m.trainIdx",m.trainIdx
            pts1.append(kp1[m.trainIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # print pts1[0], pts2[0]
    return pts1, pts2,flag

def getEssentialMat(K,pts1,pts2):
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, 1,0.90)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    u,s,vt = np.linalg.svd(F)
    s[2] = 0
    snew = np.diag(s)
    F = u @ snew @ vt
    assert np.linalg.matrix_rank(F)==2,"Rank of F not 2"

    E = K.T @ F @ K
    U,S,Vt = np.linalg.svd(E)
    S[0] = 1
    S[1] = 1
    S[2] = 0
    Snew = np.diag(S)
    E = U @ Snew @ Vt
    return E, pts1, pts2

def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def ExtractCameraPoses(E):
    poses = []

    u,s,vt = np.linalg.svd(E)
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u3 = u[:,2]
    u3 = np.reshape(u3,(3,1))
    u31 = u3
    R1 = u @ W @ vt
    if(np.linalg.det(R1)<0):
        R1 = -R1
        u31 = -u31
    P1 = np.concatenate((R1,u31), axis = 1)
    poses.append(P1)

    #P2
    u32 = u3
    R2 = u @ W @ vt
    if(np.linalg.det(R2)<0):
        R2 = -R2
        u32 = -u32
    P2 = np.concatenate((R2,-u32), axis = 1)
    poses.append(P2)

    #P3
    u33 = u3
    R3 = u @ W.T @ vt
    if(np.linalg.det(R3)<0):
        R1 = -R3
        u33 = -u33
    P3 = np.concatenate((R3,u33), axis = 1)
    poses.append(P3)

    #P4
    u34 = u3
    R4 = u @ W.T @ vt
    if(np.linalg.det(R4)<0):
        R4 = -R4
        u34 = -u34
    P4 = np.concatenate((R4,-u34), axis = 1)
    poses.append(P4)
    return poses

def LinearTriangulation(K,P0,P1,pt1,pt2):
    pt1 = np.insert(np.float32(pt1),2,1)
    pt2 = np.insert(np.float32(pt2),2,1)

    skew0 = skew(pt1)
    skew1 = skew(pt2)

    # P0 = np.concatenate((P0[:,:3], -P0[:,:3] @ P0[:,3]),axis=1)
    # P1 = np.concatenate((P1[:,:3], -P1[:,:3] @ P1[:,3]),axis=1)
    P0 = homogeneousMat(P0)
    P1 = homogeneousMat(P1)
    pose1 = K @ P0[:3,:]
    pose2 = K @ P1[:3,:]

    #Solve the equation Ax=0
    A = np.concatenate((pose1,pose2),axis=0)
    u,s,vt = np.linalg.svd(A)
    X = vt[-1]
    X = X/X[3]
    return X

def DisambiguateCameraPose(P0,poses, allPts):
    max = 0
    flag = False
    for i in range(4):
        P = poses[i]
        r3 = P[2,:3]
        r3 = np.reshape(r3,(1,3))
        C = P[:,3]
        C = np.reshape(C,(3,1))
        pts_list = allPts[i]
        pts = np.array(pts_list)
        pts = pts[:,0:3].T

        diff = np.subtract(pts,C)
        Z = r3 @ diff
        Z = Z>0
        _,idx = np.where(Z==True)
        # print(idx.shape[0])
        if max <= idx.shape[0]:
            correctPose = P
            indices = idx
            max = idx.shape[0]
    if max==0:
        flag = True
        correctPose = None
    return correctPose,flag

def homogeneousMat(P):
    t = -P[:,:3] @ P[:,3]
    # t = P[:,3]
    t = np.reshape(t,(3,1))
    P = np.concatenate((P[:,:3],t),axis = 1)
    P = np.concatenate((P,np.array([[0,0,0,1]])),axis = 0)
    return P
#########################################################################################

def main():
    # sift = cv2.xfeatures2d.SIFT_create()
    BasePath = './Oxford_dataset/stereo/centre/'
    K, LUT = getCameraMatrix('./Oxford_dataset/model')
    images = []
    P0 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])

    cam_pos = np.array([0,0,0])
    cam_pos = np.reshape(cam_pos,(1,3))
    test = os.listdir(BasePath)

    for image in sorted(test):
       # print(image)
       images.append(image)

    img1 = cv2.imread("%s/%s"%(BasePath,images[0]),0)
    # print img1.shape
    # cam_pos = np.zeros([1,2])
    # for file in range(len(images)-1):
    H1 = homogeneousMat(P0)
    for file in range(40,80):
        # print(H1,H1.shape)
        img1 = undistortImageToGray(img1,LUT)
        imgs2 = cv2.imread("%s/%s"%(BasePath,images[file+1]),0)
        img2 = undistortImageToGray(imgs2,LUT)

        pts1,pts2,flag = features(img1,img2,K)

        if ((flag==False)| (len(pts1)<=8) | (len(pts2)<=8)):
           img1 = imgs2.copy()
           print("Frame skipped")
           continue
        E,points1,points2 = getEssentialMat(K,pts1,pts2)
        # print("pts1",len(pts1))
        poses = ExtractCameraPoses(E)
        # print(len(poses))
        allPts = dict()
        for j in range(4):
           X = []
           for i in range(len(pts1)):
               pt = LinearTriangulation(K,P0,poses[j],pts1[i],pts2[i])
               X.append(pt)
           # print("Pose" + str(j))
           allPts.update({j:X})
        _,R,t,_ = cv2.recoverPose(E,pts1,pts2,K)
        print(R.shape)
        t = np.reshape(t,(3,1))
        correctPose,no_inlier = DisambiguateCameraPose(P0, poses, allPts)
        # correctPose = np.concatenate((R,t),axis=1)
        if(no_inlier):
            img1 = imgs2.copy()
            print("no inliers")
            continue
        print("Builtin",np.hstack((R,t)))
        print("Calculated",correctPose)
        P0 = correctPose
        H2 = homogeneousMat(correctPose)
        print(H1)
        x_old = H1[0,3]
        z_old = H1[2,3]
        H1 = H1 @ H2
        img1 = imgs2.copy()
        # pos = -H1[:3,:3].T @ np.reshape(H1[:3,3],(3,1))
        pos = H1[:3,3]
        print(pos.shape)
        pos = np.reshape(pos,(1,3))
        x_test = pos[0,0]
        z_test = pos[0,2]
        # print("pose shape",pos.shape)
        print("cam pose",cam_pos.shape)
        cam_pos = np.concatenate((cam_pos,pos),axis=0)
        cv2.imshow('test',img1)
        plt.plot([x_old,x_test],[z_old,z_test],'o')
        plt.pause(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.plot(cam_pos[:,0],cam_pos[:,2],".r")
    # plt.plot(h1[0,3],h1[2,3],'.b')
    plt.show()
if __name__ == '__main__':
    main()
