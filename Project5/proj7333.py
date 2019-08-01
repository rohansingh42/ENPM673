#try:
#    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#except:
#    pass

import numpy as np

import cv2
import matplotlib.pyplot as plt
import glob
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D


# In[24]:

# ###############################################################################################################
# def BuildVisibilityMatrix(nCams,im2world):
# 	wcs = np.unique(np.array(list(im2world.values())),axis=0)
# 	wc2idx = dict(zip([tuple(wc) for wc in wcs],np.arange(wcs.size,dtype=np.int32)))
# 	V = np.zeros([wcs.shape[0],nCams])
# 	x = np.zeros([wcs.shape[0],nCams,2])
# 	for ((camidx,u,v),wc) in im2world.items():
# 		row = wc2idx[tuple(wc)]
# 		V[row,camidx-1] = 1
# 		x[row,camidx-1,0] = u
# 		x[row,camidx-1,1] = v
# 	return V,wcs,x
#
# def R2r(R):
# 	out,_ = Rodrigues(R,np.zeros(3))
# 	return out
#
# def r2R(r):
# 	out,_ = Rodrigues(r,np.zeros([3,3]))
# 	return out
#
# def r2q(r):
# 	robj = Rot.from_rotvec(r)
# 	return robj.as_quat()
#
# def q2R(q):
# 	robj = Rot.from_quat(q)
# 	return r2R(robj.as_rotvec())
#
# def R2q(R):
# 	return r2q(R2r(R))
#
# def Cam2sba(n, CRCs, K):
# 	# n - number of cameras
# 	sbainp = np.zeros((n, 17))
#
# 	sbainp[:,0] = K[0][0] # fx
# 	sbainp[:,1] = K[0][2] # cx
# 	sbainp[:,2] = K[1][2] # cy
# 	sbainp[:,3] = 1       # AR
# 	sbainp[:,4] = K[0][1] # s
#
# 	for i in range(n):
# 		R = (CRCs[:,:,i])[:,:3]
# 		C = (CRCs[:,:,i])[:,3]
# 		sbainp[i,10:14] = R2q(R)
# 		sbainp[i,14:] = C.reshape((1,3))
#
# 	return sbainp
#
# def sba2Cam(newcams):
# 	pdb.set_trace()
# 	CRCs = np.zeros((3,4,newcams.shape[0]))
# 	for i in range(newcams.shape[0]):
# 		R = q2R(newcams[i,10:14])
# 		C = newcams[i,14:]
# 		CRCs[:,:3,i] = R
# 		CRCs[:,3,i] = C
#
# 	return CRCs
#
# def BundleAdjustment(X,x,K,CRCs,V):
# 	# X is 3xN, x is 2xN, CKs is 3x3xI, CRCs is 3x4xI, V is IxN
# 	pts = sba.Points(X,x,V)
# 	cams = sba.Cameras.fromDylan(Cam2sba(CRCs.shape[2], CRCs, K))
# 	newcams, newpts, info = sba.SparseBundleAdjust(cams,pts)
# 	newcams = sba2Cam(toDyl(newcams))
# 	newpts = newpts._getB()
# 	return newcams, newpts, info
#
# def toDyl(cams):
# 	result = np.zeros((cams.ncameras, 17),dtype=np.double)
# 	result[:,0:10] = cams.camarray[:,0:10]
# 	result[:,-6:] = cams.camarray[:,-6:]
#
# 	# construct missing q0 real part of unit quaternions
# 	for cam in range(cams.ncameras):
# 		result[cam,-7] = np.sqrt(1-cams.camarray[cam,-6]**2.
#                                  -cams.camarray[cam,-5]**2.
#                                  -cams.camarray[cam,-4]**2.)
#
# 	return result

###############################################################################################################
def ExtractCameraPose(E):
    poses=list()
    U,S,V=np.linalg.svd(E)
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u3=U[:,2]
    u3=u3.reshape((3,1))
    R1=np.matmul(U,np.matmul(W,V))
    if(np.linalg.det(R1)<0):
        R1=-R1
        u3=-u3
    print("P1 determinant is: ",np.linalg.det(R1))
    P1=np.hstack((R1,np.matmul(-R1,u3)))
    #LinearTriangulation(P0,P1,pts1,pts2)
    poses.append(P1)
    R2=np.matmul(U,np.matmul(W.transpose(),V))
    if(np.linalg.det(R2)<0):
        R2=-R2
        u3=-u3
    print("P2 determinant is: ",np.linalg.det(R2))
    P2=np.hstack((R2,np.matmul(-R2,u3)))
    poses.append(P2)

    R3=np.matmul(U,np.matmul(W,V))
    if(np.linalg.det(R3)<0):
        R3=-R3
        u3=-u3
    print("P3 determinant is: ",np.linalg.det(R3))
    P3=np.hstack((R3,np.matmul(-R3,-u3)))
    poses.append(P3)

    R4=np.matmul(U,np.matmul(W.transpose(),V))
    if(np.linalg.det(R4)<0):
        R4=-R4
        u3=-u3
    print("P4 determinant is: ",np.linalg.det(R4))
    P4=np.hstack((R4,np.matmul(R4,-u3)))
    poses.append(P4)
    print(P1)
    print(P2)
    print(P3)
    print(P4)
    return poses


def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def LinearTriangulation(K,pose1,pose2,pts0,pts1):
    #pose1 - pose of the first camera in the scene wrt camera 1(origin)
    #pose2 - pose of the second camera in the scene wrt camera 1(origin)
    #pts1  - image points of the same point(3D) in first camera image
    #pts2  - image points of the same point(3D) in second camera image
#     appendOne=np.zeros((np.shape(pts0)[0],1))

    pts0=np.hstack((pts0,1))
    pts1=np.hstack((pts1,1))
#     /
#     pts0=pts0[0]
#     pts1=pts1[0]

    firstPointSkew=skew(pts0)
    secondPointSkew=skew(pts1)
    pose1=np.matmul(K,pose1)
    pose2=np.matmul(K,pose2)

    firstStack=np.matmul(firstPointSkew,pose1)
    secondStack=np.matmul(secondPointSkew,pose2)

    poseStack=np.vstack((firstStack,secondStack))

    U,S,V=np.linalg.svd(poseStack)
    point=V[-1]
    point=point/point[3]
    # point[2] = np.absolute(point[2])
    #print(point)
    return point




def getInlierRANSAC(S,K):
    # S - (ptsOfimage1,ptsOfimage2)
    # K - [Calibration matrix]
    pts1=S[:,0:2]
    pts2=S[:,2:4]
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,1,0.90)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    #F should always be of rank 2
    U,S,V=np.linalg.svd(F)
    S[2]=0
    Snew=np.diag(S)
    F=np.matmul(np.matmul(U,Snew),V)
    assert np.linalg.matrix_rank(F)==2, "Rank not converging to 2, even after Forcing"
    #get the Essential Matrix from the Fundamental matrix
    E=np.matmul(K.transpose(),np.matmul(F,K))
    #E should always be of rank 2, so forcing that condition
    U,S,V=np.linalg.svd(E)
    S[0]=1
    S[1]=1
    S[2]=0
    Snew=np.diag(S)
    E=np.matmul(np.matmul(U,Snew),V)
   # ExtractCameraPose(E,pts1,pts2,imageNo,Ematrix,K)
    stacked=np.hstack((pts1,pts2))
    return E,stacked



def nonLinearFunctionTriangulation(X,P1,P2,x1,x2,K):
    X=np.reshape(X,(np.shape(X)[0],1))
    P1=np.matmul(K,P1)
    P2=np.matmul(K,P2)
    sumError = []
    X0 = np.insert(X,3,1)
    Xpt = np.reshape(X0,(4,1))
    err1 = (x1[0] - (np.dot(P1[0],Xpt)/np.dot(P1[2],Xpt)))**2 + (x1[1] - (np.dot(P1[1],Xpt)/np.dot(P1[2],Xpt)))**2
    err2 = (x2[0] - (np.dot(P2[0],Xpt)/np.dot(P2[2],Xpt)))**2 + (x2[1] - (np.dot(P2[1],Xpt)/np.dot(P2[2],Xpt)))**2

    err=err1+err2
    return err



def NonlinearTriangulation(K,pose1,pose2,pts0,pts1,X):
    #print("outside function",X[0])
    Xinit=np.reshape(X,(np.shape(X)[0],))
    filteredPoints=least_squares(nonLinearFunctionTriangulation,
                      x0=Xinit,
                     args=(pose1,pose2,pts0,pts1,K))
    return filteredPoints


def DisambiguateCameraPose(firstCamPose,poses,points):
    max=0
    for i in range(4):
        pose2=poses[i]
        r3=pose2[2,:3]
        r3=r3.reshape(1,3)
        C=pose2[:,3]
        C=C[0:3]
        C=C.reshape(3,1)
        C = np.matmul(-pose2[:,:3].T,C)
        # print(C.shape)
        # for j in range(len(allPts)):
        point=points[i]
        point = np.array(point)
        point = point[:]
        point = point[:,0:3].T
        # print("array",point)
        # diff1 = point-C
        diff = np.subtract(point,C)
        Z2=np.matmul(r3,diff)
        Z2 = Z2>0
        _,idx = np.where(Z2==True)
        if max<idx.shape[0]:
            poseid = i
            indices = idx
            max = idx.shape[0]
            correct_pose = pose2
        print(idx.shape)
    print("positive pts",max)
    return indices,correct_pose,poseid

def gen_id(u,v,diff):
    return str(np.round(u)*(10000000+diff)+np.round(v))

def reproj(X,x,P):
    sum  =0
    # print(X.shape[0])

    for pt in range(X.shape[0]):
        #Calculate reprojection error for each correspondence
        Xpt = np.insert(X[pt],3,1)
        Xpt = np.reshape(Xpt,(4,1))
        err = (x[pt,0] - (np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))**2 + (x[pt,1] - (np.dot(P[1],Xpt)/np.dot(P[2],Xpt)))**2
        sum = sum +np.sqrt(err)
    return sum/X.shape[0]

def return_corr(img,data_mat):
# for img in range(2, Nimages):
    X = []
    x = []
    corr_list = data_mat[img-2].values()
#     print(len(corr_list))
    for i in range(len(corr_list)):
        if (corr_list[i][-3] is not None) and (corr_list[i][img] is not None):
            X.append([corr_list[i][-3][0],corr_list[i][-3][1],corr_list[i][-3][2]])
            x.append([corr_list[i][img][0],corr_list[i][img][1]])
#     print(len(X),len(x))
    X = np.array(X)
    x = np.array(x)
#     print(X.shape,x.shape)
    return X,x


def LinearPnP(X,x,K):
    homo_x = np.insert(x, 2, 1,axis =1).T
    norm_x = np.matmul(np.linalg.inv(K),homo_x)
    norm_x = norm_x/norm_x[2]
    norm_x = norm_x.T
    #print(norm_x)

    A = np.zeros([1,12])

    for pt in range(x.shape[0]):
        mat = np.array([[-X[pt][0], -X[pt][1], -X[pt][2], -1,0,0,0,0, norm_x[pt][0]*X[pt][0],norm_x[pt][0]*X[pt][1], norm_x[pt][0]*X[pt][2],
                         norm_x[pt][0]*1 ],
                        [0,0,0,0,-X[pt][0],-X[pt][1],-X[pt][2],-1, norm_x[pt][1]*X[pt][0],norm_x[pt][1]*X[pt][1], norm_x[pt][1]*X[pt][2],norm_x[pt][1]*1]])
                        # [-norm_x[pt][1]*X[pt][0], -norm_x[pt][1]*X[pt][1], -norm_x[pt][1]*X[pt][2],
                        # -norm_x[pt][1]*1, norm_x[pt][0]*X[pt][0], norm_x[pt][0]*X[pt][1], norm_x[pt][0]*X[pt][2], norm_x[pt][0]*1,0,0,0,0  ]])

        #print("mat shape",mat.shape)
        A = np.concatenate((A,mat),axis=0)
    A = A[1:,:]
    #print("A matrix",A)
    U,S,Vh = np.linalg.svd(A)
    P = Vh[-1]
    #print("svd V",Vh)
    P = np.reshape(P,(3,4))
    # P = P.T
    # print(P)
    Rnew = P[:,:3]
    u,s,vh = np.linalg.svd(Rnew)
    Rnew = np.matmul(u,vh)

    Cnew = np.matmul(-np.linalg.inv(Rnew),P[:,3])
    if np.linalg.det(Rnew)<0:
        # print(np.linalg.det(Rnew))
        Rnew = -Rnew
        Cnew = -Cnew
    #     print('R',Rnew.shape)
    return Cnew, Rnew

def PnPRansac(X,x,K):
    N = 1000
    thresh = 50
    th = 0.8
    max_inliers=[]
    #Generate random indexes
    for i in range(N):
        # print(i)
        pts = [np.random.randint(0,X.shape[0]) for i in range(6)]
        Xk = np.array([ [X[pts[0],0], X[pts[0],1],X[pts[0],2]],
                        [X[pts[1],0], X[pts[1],1],X[pts[1],2]],
                        [X[pts[2],0], X[pts[2],1],X[pts[2],2]],
                        [X[pts[3],0], X[pts[3],1],X[pts[3],2]],
                        [X[pts[4],0], X[pts[4],1],X[pts[4],2]],
                        [X[pts[5],0], X[pts[5],1],X[pts[5],2]] ])
#         print(Xk.shape)
        xk = np.array([ [x[pts[0],0], x[pts[0],1]],
                        [x[pts[1],0], x[pts[1],1]],
                        [x[pts[2],0], x[pts[2],1]],
                        [x[pts[3],0], x[pts[3],1]],
                        [x[pts[4],0], x[pts[4],1]],
                        [x[pts[5],0], x[pts[5],1]] ])
#         print("xk",xk.shape)
        Cnew,Rnew = LinearPnP(Xk,xk,K)
        t = np.array([[1,0,0,-Cnew[0]],
                      [0,1,0,-Cnew[1]],
                      [0,0,1,-Cnew[2]] ])
        P = np.matmul(K,np.matmul(Rnew,t))
        Xpt = np.insert(Xk[0],3,1)
        Xpt = np.reshape(Xpt,(4,1))
        inliers = []
        sum  =0
        for pt in range(X.shape[0]):
            #Calculate reprojection error for each correspondence
            Xpt = np.insert(X[pt],3,1)
            Xpt = np.reshape(Xpt,(4,1))
            # print(Xpt.shape)

            # err1 = (x1[0] - (np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))**2 + (x1[1] - (np.dot(P1[1],Xpt)/np.dot(P1[2],Xpt)))**2

            # print("dot",x[pt,0],(np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))
            # print(xk[0,0],(np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))
            err = (x[pt,0] - (np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))**2 + (x[pt,1] - (np.dot(P[1],Xpt)/np.dot(P[2],Xpt)))**2
            sum = sum +err
            if err < thresh:
                inliers.append(pt)


            if len(max_inliers) < len(inliers):
                max_inliers = inliers
                Cnew = Cnew
                Rnew = Rnew
                # print("Inliers pnp",len(max_inliers))
                # print("Error ",err)
                if len(max_inliers) > th*X.shape[0]:
                    print("Ransac success")
                    return Cnew, Rnew, max_inliers
        # print(sum/X.shape[0])
    return Cnew, Rnew, max_inliers

def fun(P,X,x,K):
    P = np.reshape(P,(3,4))
    P = np.matmul(K,P)
    sum = 0
    for pt in range(X.shape[0]):
        Xpt = np.insert(X[pt],3,1)
        Xpt = np.reshape(Xpt,(4,1))
#         print(Xpt,x[pt,0])
        err = (x[pt,0] - (np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))**2 + (x[pt,1] - (np.dot(P[1],Xpt)/np.dot(P[2],Xpt)))**2
        # print(x[pt,0],float(np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))
#         print('\n')
        sum = sum +err
#     print(sum)
    return sum
def nonlinearPnP(X,x,K,Cnew,Rnew):

    t = np.array([[1,0,0,-Cnew[0]],
                  [0,1,0,-Cnew[1]],
                  [0,0,1,-Cnew[2]] ])
#     print(t, Rnew)
    P = np.matmul(Rnew,t)
    P_init = np.reshape(P,(P.shape[0]*P.shape[1],))
    res = least_squares(fun, P_init, args = (X,x,K))
    P_refined = res.x
    cost = res.cost
    P_refined = np.reshape(P_refined, (3,4))

    return P_refined



K = np.array([[568.996140852,0 ,643.21055941],
     [0, 568.988362396 ,477.982801038],
     [0, 0, 1]])
pathList= glob.glob('./Data/matching*.txt')
pathList.sort()
#print(pathList)
#a matrix of dimensions (6,6) that stores corresponding values of the F matrix
# data_mat = dict()
Num_images=6
StoAllmatrix=[[None for i in range(7)] for j in range(7)]
for j,path in enumerate(pathList):
    S12=np.empty((1,4))
    S13=np.empty((1,4))
    S14=np.empty((1,4))
    S15=np.empty((1,4))
    S16=np.empty((1,4))
#     txt_file_name = [i for i in path.split(".") ]
#     txt_file_name = txt_file_name[0]
#     txt_file_num = int(txt_file_name[-1])
#     dd = dict()
    with open(path) as f:
        lines=f.readlines()
        first_line = [int(s) for s in lines[0].split() if s.isdigit()]
        Num_features = int(first_line[0])
        #Initializing correspondence matrix

        lines=lines[1:]
        for feature_row,line in enumerate(lines):
#             corr_mat = [None,None,None,None,None,None,None,None,None]
            row = np.fromstring(line, dtype=float, sep=' ')
            for i in range(int(row[0])-1):
                if(row[6+i*3]==2):
                    S12=np.vstack((S12,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])

                if(row[6+i*3]==3):
                    S13=np.vstack((S13,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])

                if(row[6+i*3]==4):
                    S14=np.vstack((S14,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])

                if(row[6+i*3]==5):
                    S15=np.vstack((S15,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])

                if(row[6+i*3]==6):
                    S16=np.vstack((S16,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
            #dd[idd] = corr_mat

        #print(S16.shape)
        S12=S12[1:]
        S13=S13[1:]
        S14=S14[1:]
        S15=S15[1:]
        S16=S16[1:]
        print(j)
        k=0
        if(S12.shape[0]!=0):
            _,cleanPoints=getInlierRANSAC(S12,K)
            k+=1
            StoAllmatrix[j+1][j+k+1]=cleanPoints
        if(S13.shape[0]!=0):
            _,cleanPoints=getInlierRANSAC(S13,K)
            k+=1
            StoAllmatrix[j+1][j+k+1]=cleanPoints
        if(S14.shape[0]!=0):
            _,cleanPoints=getInlierRANSAC(S14,K)
            k+=1
            StoAllmatrix[j+1][j+k+1]=cleanPoints
        if(S15.shape[0]!=0):
            _,cleanPoints=getInlierRANSAC(S15,K)
            k+=1
            StoAllmatrix[j+1][j+k+1]=cleanPoints
        if(S16.shape[0]!=0):
            _,cleanPoints=getInlierRANSAC(S16,K)
            k+=1
            StoAllmatrix[j+1][j+k+1]=cleanPoints
        #data_mat.update({j:d})

        #data_mat.append(corr_mat)
#print(StoAllmatrix[1][2])
#we need to get the E matrix between the first two images
E,_=getInlierRANSAC(StoAllmatrix[1][2],K)

#now decompose and this E matrix and extract all the possible poses
firstPoses=ExtractCameraPose(E)

#loop over the first scene poses and get the 3D triangulated points
tmp=np.zeros((3,1))
firstCamPose=np.hstack((np.identity(3),tmp))
pts=StoAllmatrix[1][2]
pts1All=pts[:,0:2]
pts2All=pts[:,2:4]

pts1=pts1All
pts2=pts2All
# print(pts1)
# print(pts2)
allPts = []
for i in range(4):
    worldPoints=[]
    for j in range(0,pts1All.shape[0]):
        point=LinearTriangulation(K,firstCamPose,firstPoses[i],pts1[j],pts2[j])
        # print(point)
        worldPoints.append(point)
    allPts.append(worldPoints)
ids, correctedPose,poseid=DisambiguateCameraPose(firstCamPose,firstPoses,allPts)
print(correctedPose)
worldPointsLinear = np.array(allPts[poseid])
worldPointsLinear = worldPointsLinear[:]
worldPointsLinear = worldPointsLinear[ids]
pts1=pts1All[ids]
pts2=pts2All[ids]
print(worldPointsLinear.shape)
error1 = reproj(worldPointsLinear[:,:3],pts2,np.matmul(K,correctedPose))
print("Reproj Error After Linear Triangulation",error1)
np.save("worldPointsLinear.npy",worldPointsLinear)





#plt.plot(worldPointsLinear[:,0],worldPointsLinear[:,2],'.b')
# plt.show()
# worldPointsLinear=np.zeros((1,4))
# for i in range(0,pts1All.shape[0]):
# /    point=LinearTriangulation(K,firstCamPose,correctedPose,pts1All[i],pts2All[i])
    # point=point.reshape((4,1))
    # worldPointsLinear=np.vstack((worldPointsLinear,point.T))
#
# pathList= glob.glob('./Data/matching*.txt')
# pathList.sort()
# #print(pathList)
# #a matrix of dimensions (6,6) that stores corresponding values of the F matrix
# data_mat = dict()
# Num_images=6
# for j,path in enumerate(pathList):
#     txt_file_name = [i for i in path.split(".") ]
#     txt_file_name = txt_file_name[0]
#     txt_file_num = int(txt_file_name[-1])
#     dd = dict()
#     with open(path) as f:
#         lines=f.readlines()
#         first_line = [int(s) for s in lines[0].split() if s.isdigit()]
#         Num_features = int(first_line[0])
#         #Initializing correspondence matrix
#         lines=lines[1:]
#         for feature_row,line in enumerate(lines):
#             corr_mat = [None,None,None,None,None,None,None,None,None]
#             row = np.fromstring(line, dtype=float, sep=' ')
#
#             idd = gen_id(row[4],row[5],txt_file_num)
#             if idd in dd:
#                 # print('skipping points')
#
#                 if(int(row[0]) <= int(dd[idd][-1])):
#                     continue
#
#             corr_mat[txt_file_num-1]=(row[4],row[5])
#             corr_mat[-2] = (row[1],row[2],row[3])
#             corr_mat[-1] = int(row[0])
#             for i in range(int(row[0])-1):
#                 if(row[6+i*3]==2):
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
#
#                 if(row[6+i*3]==3):
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
#
#                 if(row[6+i*3]==4):
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
#
#                 if(row[6+i*3]==5):
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
#
#                 if(row[6+i*3]==6):
#                     corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
#
#             dd[idd] = corr_mat
#         data_mat.update({j:dd})
#
#
#
#
#
################################################################################################################################
pathList= glob.glob('./Data/matching*.txt')
pathList.sort()
#print(pathList)
#a matrix of dimensions (6,6) that stores corresponding values of the F matrix
data_mat = dict()
# d = dict()

# StoAllmatrix=[[None for i in range(7)] for j in range(7)]
for j,path in enumerate(pathList):
    # S12=np.empty((1,4))
    # S13=np.empty((1,4))
    # S14=np.empty((1,4))
    # S15=np.empty((1,4))
    # S16=np.empty((1,4))
    txt_file_name = [i for i in path.split(".") ]
    txt_file_name = txt_file_name[1]
    txt_file_num = int(txt_file_name[-1])
    d = dict()
    with open(path) as f:
        lines=f.readlines()
        first_line = [int(s) for s in lines[0].split() if s.isdigit()]
        Num_features = int(first_line[0])
        lines=lines[1:]
        for feature_row,line in enumerate(lines):
            corr_mat = [None,None,None,None,None,None,None,None,None]
            row = np.fromstring(line, dtype=float, sep=' ')

            idd = gen_id(row[4],row[5],txt_file_num)
            if idd in d:
                # print('skipping points')

                if(int(row[0]) <= int(d[idd][-1])):
                    continue

            corr_mat[txt_file_num-1]=(row[4],row[5])
            corr_mat[-2] = (row[1],row[2],row[3])
            corr_mat[-1] = int(row[0])
            for i in range(int(row[0])-1):

                if(row[6+i*3]==2):
                    # S12=np.vstack((S12,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
                    corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
                if(row[6+i*3]==3):
                    # S13=np.vstack((S13,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
                    corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
                if(row[6+i*3]==4):
                    # S14=np.vstack((S14,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
                    corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
                if(row[6+i*3]==5):
                    # S15=np.vstack((S15,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
                    corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])
                if(row[6+i*3]==6):
                    # S16=np.vstack((S16,np.array([row[4],row[5],row[6+i*3+1],row[6+i*3+2]])))
                    corr_mat[int(row[6+i*3])-1]=(row[6+i*3+1],row[6+i*3+2])

            d[idd] = corr_mat

        data_mat.update({j:d})
#
#
# worldPointsLinear=worldPointsLinear[1:]
world=np.zeros((1,3))
for i in range(np.shape(worldPointsLinear)[0]):
    points=NonlinearTriangulation(K,firstCamPose,correctedPose,pts1[i],pts2[i],worldPointsLinear[i,:3])
    worldPointsNL=points.x
    point=worldPointsNL
    point=np.insert(point,3,1)
    worldPointsNL.shape[0]
    worldPointsNL=np.reshape(worldPointsNL,(np.shape(worldPointsNL)[0]/3,3))
    idd=None
    idd = gen_id(pts1All[i][0],pts1All[i][1],1)
    first_dict = data_mat[0]
    if idd in first_dict:
        list_corr_id = first_dict.get(idd)

    point=point.reshape((4,1))
    curr_world = np.reshape(point[:-1],(point.shape[0]-1,))
    list_corr_id[-3]=tuple(curr_world)
    data_mat[0][idd]=list_corr_id
    #print(worldPointsNL)
    world=np.vstack((world,worldPointsNL))
world=world[1:]
error2 = reproj(world,pts2,np.matmul(K,correctedPose))
print("Reprojection error after nonlinear Triangulation",error2)
np.save("worldpointsNonlinear.npy",world)
#

# Ctest,Rtest=LinearPnP(world,pts2,K)
# t = np.array([[1,0,0,-Ctest[0]],
#               [0,1,0,-Ctest[1]],
#               [0,0,1,-Ctest[2]] ])
# P = np.matmul(Rtest,t)
# print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# print(correctedPose)
# print(P)
#



#################################################################################################################################
# plt.plot(world[:,0],world[:,2],'.r')
# plt.xlim([-50,100])
# plt.ylim([0,100])
# plt.show()
#
Nimages = 6
P = correctedPose
l = []
dist = np.array([0,0,0,0,0])
dist = np.reshape(dist,(1,5))
# print(StoAllmatrix)
for img in range(2,Nimages):
    # img = 2
    X,x = return_corr(img, data_mat)
    Cnew, Rnew, inliers = PnPRansac(X,x,K)
    # _,rvecs, tvecs, inliers  = cv2.solvePnPRansac(X, x, K, dist)
    # print("My ransac",Cnew,Rnew)
    # print("cv2  ransac",tvecs,rvecs)
    tz = np.array([[1,0,0,-Cnew[0]],
                  [0,1,0,-Cnew[1]],
                  [0,0,1,-Cnew[2]] ])
#     print(t, Rnew)
    Pz = np.matmul(Rnew,tz)
    res = nonlinearPnP(X,x,K,Cnew,Rnew)
    #res=np.matmul(np.linalg.inv(K),res)
    worldPointsPnP=np.zeros((1,4))
    pts=StoAllmatrix[img][img+1]
    pts2All=pts[:,0:2]
    pts3All=pts[:,2:4]
    for i in range(0,pts2All.shape[0]):
        point=LinearTriangulation(K,P,res,pts2All[i],pts3All[i])
        point=point.reshape((4,1))
        points=NonlinearTriangulation(K,P,res,pts2All[i],pts3All[i],point[0:3])
        worldPointsNL=points.x
        point=worldPointsNL
        point=np.insert(point,3,1)
#        worldPointsNL=np.reshape(worldPointsNL,(np.shape(worldPointsNL)[0]/3,3))
        #point=point/point[3]
        worldPointsPnP=np.vstack((worldPointsPnP,point.T))
        #Save 3D points for each Triangulation
        # print(pts2All[i][0],pts2All[i][1])
        idd = gen_id(pts2All[i][0],pts2All[i][1],img)
        curr_elem = data_mat[img-1][idd]
        point=point.reshape((4,))
        curr_elem[-3] = tuple(point[:3])
        # print(curr_elem)
        data_mat[img-1][idd] = curr_elem

    P = res
    worldPointsPnP=worldPointsPnP[1:]
    error3 = reproj(world,pts3All,np.matmul(K,P))
    print("Reprojection error after nonlinear Triangulation"+str(img+1),error3)
    path = 'reprojection error' + str(img+1) + '.npy'
    np.save(path, worldPointsPnP)

    l.append(worldPointsPnP)
    if img == 2:
        print("P3 found")
    elif img == 3:
        print("P4 found")
    elif img == 4:
        print("P5 found")
    elif img == 5:
        print("P6 found")
# #
# # # plt.rcParams['figure.figsize']=[10,10]
# # # plt.plot(worldPointsLinear[:,0],worldPointsLinear[:,2],'.r')
# # # plt.plot(worldPointsPnP[:,0],worldPointsPnP[:,2],'.g')
# # #
# # # # plt.show()
# # # plt.show(
np.save('wrldpts.npy',world)
np.save('scatter.npy',l)
# np.save('Poses.npy',P)
# #
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(0, 0, 0, zdir='y', s=20, c='magenta')
ax.scatter(world[:,0], world[:,1], world[:,2],c='b',marker='.', zdir='y')
ax.scatter(l[0][:,0],l[0][:,1],l[0][:,2],c='g',marker='.',zdir='y')
ax.scatter(l[1][:,0],l[1][:,1],l[1][:,2],c='m',marker='.',zdir='y')
ax.scatter(l[2][:,0],l[2][:,1],l[2][:,2],c='y',marker='.',zdir='y')
ax.scatter(l[3][:,0],l[3][:,1],l[3][:,2],c='c',marker='.',zdir='y')

ax.set_xlim([-50,50])
ax.set_ylim([-50,50])
ax.set_zlim([-50,50])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
#print(data_mat[0][idd][-3])