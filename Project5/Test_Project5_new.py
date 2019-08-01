import sys,os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('/opt/ros/kinetic/lib/python3.3/dist-packages')
except:
    pass

import cv2
import numpy as np
import glob
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import matplotlib.pyplot as plt
import os

np.set_printoptions(suppress=True)

def ExtractCameraPose(E):
    W = np.array(([0,-1,0],[1,0,0],[0,0,1]))
    U,S,V = np.linalg.svd(E)
    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]
    R1 = U @ W @ V
    R2 = U @ W @ V
    R3 = U @ W.T @ V
    R4 = U @ W.T @ V
    
    if np.linalg.det(R1)==-1:
        R1=-1*R1
        
    if np.linalg.det(R2)==-1:
        R2=-1*R2
        
    if np.linalg.det(R3)==-1:
        R3=-1*R3
        
    if np.linalg.det(R4)==-1:
        R4=-1*R4
        

    return C1,C2,C3,C4,R1,R2,R3,R4


def get_best_pair(K,C1,R1,C2,R2,U,V):
    
    H1=np.hstack((R1,C1))
    H1=np.vstack((H1,[0,0,0,1]))
    #H1=K_new@H1
    
    H2=np.hstack((R2,C2))
    H2=np.vstack((H2,[0,0,0,1]))
    #H2=K_new@H2

    x_new=[]
    for i in range(0,np.shape(U)[0]):

        Anew=np.vstack((U[i][0]*H1[2,:]-H1[0,:],
                        U[i][1]*H1[2,:]-H1[1,:],
                        V[i][0]*H2[2,:]-H2[0,:],
                        V[i][1]*H2[2,:]-H2[1,:]))

        _,_,v=np.linalg.svd(Anew)
        X=v[:,3]
        X=X/X[3]
        X=X[0:3]

        x_new.append(X)
    return x_new

def DisambiguateCameraPose(C_set,R_set,X_set):

    check_temp=[]
    count=0
    
    for i in range(0,4):
        new_count=0
        for j in range(0,X1.shape[0]):
            x_temp=np.zeros((3,1))
            x_temp[0]=X_set[i][j][0]
            x_temp[1]=X_set[i][j][1]
            x_temp[2]=X_set[i][j][2]
            H= np.hstack((R_set[i],C_set[i]))
            H=np.vstack((H,[0,0,0,1]))
            if R_set[i][2,:]@(x_temp[i][j,:]-C_set[i]):
                new_count+=1

        check_temp.append(new_count)

        index=np.where(check_temp==np.amax(check_temp))
    C_set=C_set[index[0][0]]
    R_set=R_set[index[0][0]]
    X_set=X_set[index[0][0]]
    return C_set,R_set,X_set

# #def randomMatchingPoints(matchpts1,matchpts2):
# #    rand_index = np.random.randint(len(matchpts1), size=8)
# #    
# #    X1 = np.array([matchpts1[rand_index[0]],matchpts1[rand_index[1]],matchpts1[rand_index[2]],matchpts1[rand_index[3]],matchpts1[rand_index[4]],matchpts1[rand_index[5]],matchpts1[rand_index[6]],matchpts1[rand_index[7]]]) 
# #    X2 = np.array([matchpts2[rand_index[0]],matchpts2[rand_index[1]],matchpts2[rand_index[2]],matchpts2[rand_index[3]],matchpts2[rand_index[4]],matchpts2[rand_index[5]],matchpts2[rand_index[6]],matchpts2[rand_index[7]]]) 
# #    
# #    return X1, X2
    
# def randomMatchingPoints(matchpts1,matchpts2):
    
#     reso = (960/8, 1280/8)
#     grid_set = {}
#     dummy = 0
#     for i in range(8):
#         for j in range(8):
#             xs,xe = i*reso[1], (i+1)*reso[1]
#             ys,ye = i*reso[0], (i+1)*reso[0]
#             index_grid = np.where((matchpts1[:,0]<xe) & (matchpts1[:,0]>xs) & (matchpts1[:,1]>ys) & (matchpts1[:,1]<ye))
            
#             if index_grid[0].size:
#                 grid_set[dummy] = index_grid[0]
#                 dummy+=1

#     index_set = np.random.randint(len(grid_set), size=8)
#     index = []
#     for k in index_set:
#         t = grid_set[k]
#         index.append(t[np.random.randint(len(t))])      
        
#     X1 = np.array([matchpts1[index[0]],matchpts1[index[1]],matchpts1[index[2]],matchpts1[index[3]],matchpts1[index[4]],matchpts1[index[5]],matchpts1[index[6]],matchpts1[index[7]]]) 
#     X2 = np.array([matchpts2[index[0]],matchpts2[index[1]],matchpts2[index[2]],matchpts2[index[3]],matchpts2[index[4]],matchpts2[index[5]],matchpts2[index[6]],matchpts2[index[7]]])
  
#     return X1, X2


# def epipolar_distance(F,X):
#     l = F @ X
#     d = (l[0]*X[0] + l[1]*X[1] + l[2])/np.sqrt(l[0]**2 + l[1]**2)
    
#     return d



centre_path = glob.glob("Oxford_dataset/stereo/centre/*.png")
fx, fy, cx, cy, G_camera_image, LUT=ReadCameraModel("Oxford_dataset/model")
K = np.zeros((3,3))
K[0][0] = fx
K[1][1] = fy
K[2][2] = 1
K[0][2] = cx
K[1][2] = cy

N = len(centre_path)
camera_centre=np.zeros((4,1))
camera_centre[3]=1
new_centre=[]
count_final=0
H_new=np.identity(4)
C_origin=np.zeros((3,1))
R_origin=np.identity(3)


for i in range(0,N):
    img_current_frame = cv2.imread(centre_path[i], 0)
    img_next_frame = cv2.imread(centre_path[i+1], 0)
    img_current_frame= cv2.rectangle(img_current_frame,(np.float32(50),np.float32(np.shape(img_current_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
    img_next_frame= cv2.rectangle(img_next_frame,(np.float32(50),np.float32(np.shape(img_next_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
    color_img = cv2.cvtColor(img_current_frame, cv2.COLOR_BayerGR2RGB)
    color_img_next = cv2.cvtColor(img_next_frame, cv2.COLOR_BayerGR2RGB)
    
    undist_img = UndistortImage(color_img,LUT)
    undist_img_next = UndistortImage(color_img_next,LUT)
    MIN_MATCH_COUNT = 10
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(undist_img,None)
    kp2, des2 = sift.detectAndCompute(undist_img_next,None)
    
        
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    U = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    V = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    E,_=cv2.findEssentialMat(U,V,focal=fx,pp=(cx,cy),method=cv2.RANSAC,prob=0.999,threshold=0.5)
    
    C1,C2,C3,C4,R1,R2,R3,R4=ExtractCameraPose(E)
    C1=(C1).reshape(-1,1)
    C2=(C2).reshape(-1,1)
    C3=(C3).reshape(-1,1)
    C4=(C4).reshape(-1,1)

    # C_set=np.array((C1,C2,C3,C4))
    # R_set=np.array((R1,R2,R3,R4))
    U_new=np.zeros((np.shape(U)[0],3))
    U_new[:,0:2]=U
    U_new[:,2]=1
    
    V_new=np.zeros((np.shape(V)[0],3))
    V_new[:,0:2]=V
    V_new[:,2]=1
    
    X1=np.array(get_best_pair(K,C_origin,R_origin,C1,R1,U_new,V_new))
    X2=np.array(get_best_pair(K,C_origin,R_origin,C2,R2,U_new,V_new))
    X3=np.array(get_best_pair(K,C_origin,R_origin,C3,R3,U_new,V_new))
    X4=np.array(get_best_pair(K,C_origin,R_origin,C4,R4,U_new,V_new))

    C_set=np.array((C1,C2,C3,C4))
    R_set=np.array((R1,R2,R3,R4))
    X_set=np.array((X1,X2,X3,X4))
    
    C,R,X=DisambiguateCameraPose(C_set,R_set,X_set)
    _,R_new,C_new,_=cv2.recoverPose(E, U, V, focal=fx, pp=(cx,cy))
    if np.linalg.det(R)<0:
         R = -R
    print("BUILT_IN\n",R_new)
    print("MY FUCNTION\n",R)
    
    print("=========\n")
    
    H_final= np.hstack((R,C))
    H_final=np.vstack((H_final,[0,0,0,1]))

    #print("check %d"%count_final)
    count_final+=1

    x_old=H_new[0][3]
    z_old=H_new[2][3]
    H_new=H_new@H_final
    x_test=H_new[0][3]
    z_test=H_new[2][3]
    

    #camera_centre=H_new@camera_centre
    #new_centre.append(camera_centre)
    cv2.imshow('11',color_img)
    plt.plot([x_old,x_test],[-z_old,-z_test],'o')
    plt.pause(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
