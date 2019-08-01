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
import matplotlib.pyplot as plt

def warpAll(t,I,p,Igradx,Igrady):
    wI = np.zeros(t.shape)
    wIgradx = np.zeros(t.shape)
    wIgrady = np.zeros(t.shape)
    p = np.reshape(p,(6,))
    pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
    pw = np.reshape(pw,(2,3))
    # x = np.array([0,0,1]).T
    # x = np.reshape(x,(3,1))
    # # print(p.shape)
    # # print(x.shape)
    # W = np.int16(np.round(np.matmul(pw,x)))
    # print(W)
    
    for i in range(t.shape[1]):            # x
        for j in range(t.shape[0]):       # y
            x = np.array([i,j,1]).T
            x = np.reshape(x,(3,1))
            # print(p.shape)
            # print(x.shape)
            W = np.int16(np.round(np.matmul(pw,x)))
            # print(W[1])
            # print(W.shape)
            # W = np.int64(W)
            a = I[W[1],W[0]]
            igx = Igradx[W[1],W[0]]
            igy = Igrady[W[1],W[0]]
            wI[j,i] = a
            # print(wI[j,i])
            wIgradx[j,i] = igx
            wIgrady[j,i] = igy
    # cv2.imshow('wI',wI)
    # cv2.imshow('wIgradx',wIgradx)
    # cv2.imshow('wIgrady',wIgrady)
    # while True:
    #     if cv2.waitKey(00)==ord('q'):
    #         break
    # print(wI[wI == 255])
    print(wI.shape)
    # plt.imshow(wI)
    # plt.show()
    return wI, wIgradx, wIgrady


def computeP(t,I,pprev,thresh,Igradx,Igrady,pinv,xl,xu,yl,yu):
    # t = wIprev
    p = pprev
    delpnorm = thresh + 10
    # cv2.imshow('Igradx',Igradx)
    # cv2.imshow('Igrady',Igrady)
    it = np.int(0)
    while (delpnorm > thresh) & (it < 300):
        # wI, wIgradx, wIgrady = warpAll(t,I,p,Igradx,Igrady)
        
        pinv = np.array([-p[0] - p[0]*p[3] + p[1]*p[2]
                        ,-p[1]
                        ,-p[2]
                        ,-p[3] - p[0]*p[3] + p[1]*p[2]
                        ,-p[4] - p[3]*p[4] + p[2]*p[5]
                        ,-p[5] - p[0]*p[5] + p[1]*p[4]])
        pinv = (1/((1+p[0])*(1+p[3])-p[1]*p[2]))*pinv
        # print(pinv.shape)
        pinv = np.reshape(pinv,(6,1))
        pwinv = np.array([[1+pinv[0], pinv[2], pinv[4]],[pinv[1], 1+pinv[3], pinv[5]]])
        pwinv = np.reshape(pwinv,(2,3))
        wI = cv2.warpAffine(I,pwinv,(I.shape[1],I.shape[0]))
        wIgradx = cv2.warpAffine(Igradx,pwinv,(I.shape[1],I.shape[0]))
        wIgrady = cv2.warpAffine(Igrady,pwinv,(I.shape[1],I.shape[0]))
        
        wI = wI[yl:yu,xl:xu]
        wIgradx = wIgradx[yl:yu,xl:xu]
        wIgrady = wIgrady[yl:yu,xl:xu]
        # print('wI',wI.shape)
        # pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
        # pw = np.reshape(pw,(2,3))
        # plt.imshow(wI)
        # plt.show()
        s1 = np.zeros([6,1])
        s2 = np.zeros([6,6])
        for i in range(t.shape[1]):            # x
            for j in range(t.shape[0]):       # y
                wJ = np.array([[i,0,j,0,1,0],[0,i,0,j,0,1]])
                ig = np.array([wIgradx[j,i],wIgrady[j,i]])
                # print('wJ',wJ.shape)
                # print('ig',ig.shape)
                b1 = np.matmul(ig,wJ)
                b2 = t[j,i] - wI[j,i]
                # b2 = np.absolute(t[j,i] - wI[j,i])
                b1 = np.reshape(b1,(1,6))
                b2 = np.reshape(b2,(1,1))
                # print('b2',b2)
                # print('b1',b1.shape)
                # print('b2',b2.shape)
                s1 = s1 + b1.T*b2
                s2 = s2 + b1.T*b1
        H = s2
        sdpu = s1
        # print('sdpu',sdpu)
        # print(t.shape)
        # print(wI.shape)
        # print('H',H.shape)
        # print('sdpu',sdpu.shape)
        delp = np.matmul(np.linalg.inv(H),sdpu)
        # delp = -0.00000001*sdpu
        # print('delp',delp.shape)
        p = p + delp
        delpnorm = np.linalg.norm(delp)
        # print('p',p)
        # print('delp',delp)
        print('dpnorm',delpnorm)
        # while True:
        #     if cv2.waitKey(00)==ord('q'):
        #         break
        it = it + 1
    
    return p,wI

img1 = cv2.imread('./data/vase/0019.jpg')
# temp = np.zeros([60,80])
xl = 123
xu = 172
yl = 89
yu = 151
temp = img1[yl:yu,xl:xu]
temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
cv2.imwrite('./data/vase_template.jpg',temp)
p = np.zeros([6,1])
p[4] = 0
p[5] = 0
pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
pw = np.reshape(pw,(2,3))
c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
c2 = np.matmul(pw,np.reshape([xu-xl,0,1],(3,1)))
c3 = np.matmul(pw,np.reshape([xu-xl,yu-yl,1],(3,1)))
c4 = np.matmul(pw,np.reshape([0,yu-yl,1],(3,1)))
print('c1',c1)
print('c3',c3)

# cv2.rectangle(img1,(c1[0],c1[1]),(c3[0],c3[1]),(0,255,0),3)
# cv2.imshow('t',img1)
# warped_img = cv2.warpAffine(input,W_xp,(input.shape[1],input.shape[0]))
# error_img = temp - warped_img
# #Calculate Gradients
# Ix = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize =3)
# Iy = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize =3)
# Igrad = np.dstack([Ix,Iy])
# # print(Igrad.shape)
# # Jac = np.array()
# cv2.imshow('test',temp)
# if cv2.waitKey(00)==ord('q'):
#     cv2.destroyAllWindows()
# plt.imshow(img1)
# plt.show()
pprev = p
frno = 20
while True:
    if frno < 100:
        frame = cv2.imread('./data/vase/00'+str(frno)+'.jpg')
    else:
        frame = cv2.imread('./data/vase/0'+str(frno)+'.jpg')
    I = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Igradx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize =3)
    Igrady = cv2.Sobel(I,cv2.CV_64F,0,1,ksize =3)
    thresh = 0.001
    pinv = np.array([-pprev[0] - pprev[0]*pprev[3] + pprev[1]*pprev[2]
                    ,-pprev[1]
                    ,-pprev[2]
                    ,-pprev[3] - pprev[0]*pprev[3] + pprev[1]*pprev[2]
                    ,-pprev[4] - pprev[3]*pprev[4] + pprev[2]*pprev[5]
                    ,-pprev[5] - pprev[0]*pprev[5] + pprev[1]*pprev[4]])
    pinv = (1/((1+pprev[0])*(1+pprev[3])-pprev[1]*pprev[2]))*pinv
    print(pinv.shape)
    pinv = np.reshape(pinv,(6,1))

    pnew, wIprev = computeP(temp,I,pprev,thresh,Igradx,Igrady,pinv,xl,xu,yl,yu)
    print('pnew',pnew)
    pw = np.array([[1+pnew[0], pnew[2], pnew[4]],[pnew[1], 1+pnew[3], pnew[5]]])
    pw = np.reshape(pw,(2,3))
    c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
    c2 = np.matmul(pw,np.reshape([xu-xl,0,1],(3,1)))
    c3 = np.matmul(pw,np.reshape([xu-xl,yu-yl,1],(3,1)))
    c4 = np.matmul(pw,np.reshape([0,yu-yl,1],(3,1)))
    # print('c1',c1)
    # print('c3',c3)
    corners = np.int64(np.hstack([c1,c2,c3,c4]))
    print(corners)
    x1 = min(corners[0,:])
    y1 = min(corners[1,:])
    x2 = max(corners[0,:])
    y2 = max(corners[1,:])
    print(x1,y1)

    cv2.rectangle(frame,(xl+x1,yl+y1),(xl+x2,yl+y2),(0,255,0),3)
    # cv2.imshow('I',I)
    cv2.imwrite('./data/vase_modtry7/vm' + str(frno) + '.jpg',frame)

    # while True:
    #     if cv2.waitKey(00)==ord('q'):
    #         break
    if frno == 169:
        break
    pprev = pnew
    frno = frno + 1
    # temp = wIprev