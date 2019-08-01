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
    tmean = np.mean(t)
    wimean = np.mean(wI)
    wI = wI*(tmean/wimean)
    print(wI.shape)
    # plt.imshow(wI)
    # plt.show()
    return wI, wIgradx, wIgrady


def computeP(t,I,pprev,thresh,Igradx,Igrady,xl,xu,yl,yu):
    # t = wIprev
    # tmean = np.mean(t)
    # imean = np.mean(I)
    # I = I*(tmean/imean)
    p = pprev
    delpnorm = thresh + 10
    # cv2.imshow('Igradx',Igradx)
    # cv2.imshow('Igrady',Igrady)
    it = np.int(0)
    delpnormprev = 10
    while (delpnorm > thresh):
        wI, wIgradx, wIgrady = warpAll(t,I,p,Igradx,Igrady)
        
        # pinv = np.array([-p[0] - p[0]*p[3] + p[1]*p[2]
        #                 ,-p[1]
        #                 ,-p[2]
        #                 ,-p[3] - p[0]*p[3] + p[1]*p[2]
        #                 ,-p[4] - p[3]*p[4] + p[2]*p[5]
        #                 ,-p[5] - p[0]*p[5] + p[1]*p[4]])
        # pinv = (1/((1+p[0])*(1+p[3])-p[1]*p[2]))*pinv
        # # print(pinv.shape)
        # pinv = np.reshape(pinv,(6,1))
        # pwinv = np.array([[1+pinv[0], pinv[2], pinv[4]],[pinv[1], 1+pinv[3], pinv[5]]])
        # pwinv = np.reshape(pwinv,(2,3))
        # W_xp = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
        # wI = cv2.warpAffine(I,W_xp,(t.shape[1],t.shape[0]))
        # wIgradx = cv2.warpAffine(Igradx,W_xp,(t.shape[1],t.shape[0]))
        # wIgrady = cv2.warpAffine(Igrady,W_xp,(t.shape[1],t.shape[0]))
        s1 = np.zeros([6,1])
        s2 = np.zeros([6,6])
        # diff = t - wI
        k = 500
        for i in range(t.shape[1]):#xl,xu):            # x
            for j in range(t.shape[0]):#yl,yu):       # y
                wJ = np.array([[i,0,j,0,1,0],[0,i,0,j,0,1]])
                ig = np.array([wIgradx[j,i],wIgrady[j,i]])
                # print('wJ',wJ.shape)
                # print('ig',ig.shape)
                diff = t[j,i] - wI[j,i]
                if np.absolute(diff) < k:
                    weight = 1
                else:
                    weight = k/np.absolute(diff)
                b1 = np.matmul(ig,wJ)
                b2 = diff
                b1 = np.reshape(b1,(1,6))
                b2 = np.reshape(b2,(1,1))
                weight = weight*np.identity(6)
                # print(weight)
                # print('b2',b2)
                # print('b1',b1.shape)
                # print('b2',b2.shape)
                # s1 = s1 + np.multiply(b1.T*b2, weight)
                # s2 = s2 + np.multiply(b1.T*b1, weight)
                s1 = s1 + np.matmul(weight,b1.T)*b2
                s2 = s2 + np.matmul(weight, b1.T*b1)
        H = s2
        sdpu = s1
        # print('sdpu',sdpu)
        # print(t.shape)
        # print(wI.shape)
        # print('H',H)
        # print('sdpu',sdpu.shape)
        delp = np.matmul(np.linalg.inv(H),sdpu)
        # print('delp',delp.shape)
        delpnorm = np.linalg.norm(delp)
        if delpnormprev > delpnorm:
            pmin = p
        p = p + delp
        # print('p',p)
        # print('delp',delp)
        delpnormprev = delpnorm
        print('dpnorm',delpnorm)
        # while True:
        #     if cv2.waitKey(00)==ord('q'):
        #         break
        if  (it > 100):
            p = pmin
            break
        it = it + 1
    # plt.imshow(wI)
    # plt.show()
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
# temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
# input = img2[70:150,117:177]
# temp = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# temp = cv2.GaussianBlur(temp,(5,5),0)
temp = cv2.medianBlur(temp,5)
p = np.zeros([6,1])
p[4] = xl
p[5] = yl
# pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
# pw = np.reshape(pw,(2,3))
# c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
# c2 = np.matmul(pw,np.reshape([xu-xl,0,1],(3,1)))
# c3 = np.matmul(pw,np.reshape([xu-xl,yu-yl,1],(3,1)))
# c4 = np.matmul(pw,np.reshape([0,yu-yl,1],(3,1)))
# print('c1',c1)
# print('c3',c3)
# print(temp.shape)

# cv2.rectangle(img1,(c1[0],c1[1]),(c3[0],c3[1]),(0,255,0),3)
# cv2.imshow('t',img1)
# plt.imshow(img1)
# plt.show()
# if cv2.waitKey(00)==ord('q'):
#     cv2.destroyAllWindows()
pprev = p
frno = 20
while True:
    if frno < 100:
        frame = cv2.imread('./data/vase/00'+str(frno)+'.jpg')
    else:
        frame = cv2.imread('./data/vase/0'+str(frno)+'.jpg')
    I = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    I = cv2.medianBlur(I,5)
    Igradx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize =3)
    Igrady = cv2.Sobel(I,cv2.CV_64F,0,1,ksize =3)
    # I = cv2.GaussianBlur(I,(5,5),0)
    thresh = 0.005
    pnew, wIprev = computeP(temp,I,pprev,thresh,Igradx,Igrady,xl,xu,yl,yu)
    print('pnew',pnew)
    # pinv = np.array([-pnew[0] - pnew[0]*pnew[3] + pnew[1]*pnew[2]
    #                 ,-pnew[1]
    #                 ,-pnew[2]
    #                 ,-pnew[3] - pnew[0]*pnew[3] + pnew[1]*pnew[2]
    #                 ,-pnew[4] - pnew[3]*pnew[4] + pnew[2]*pnew[5]
    #                 ,-pnew[5] - pnew[0]*pnew[5] + pnew[1]*pnew[4]])
    # pinv = (1/((1+pnew[0])*(1+pnew[3]) - pnew[1]*pnew[2]))*pinv
    # # print(pinv.shape)
    # pinv = np.reshape(pinv,(6,1))
    # pw = np.array([[1+pinv[0], pinv[2], pinv[4]],[pinv[1], 1+pinv[3], pinv[5]]])
    # pw = np.reshape(pw,(2,3))
    pw = np.array([[1+pnew[0], pnew[2], pnew[4]],[pnew[1], 1+pnew[3], pnew[5]]])
    pw = np.reshape(pw,(2,3))
    # c1 = np.matmul(pw,np.reshape([xl,yl,1],(3,1)))
    # c2 = np.matmul(pw,np.reshape([xu,yl,1],(3,1)))
    # c3 = np.matmul(pw,np.reshape([xu,yu,1],(3,1)))
    # c4 = np.matmul(pw,np.reshape([xl,yu,1],(3,1)))
    c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
    c2 = np.matmul(pw,np.reshape([xu-xl,0,1],(3,1)))
    c3 = np.matmul(pw,np.reshape([xu-xl,yu-yl,1],(3,1)))
    c4 = np.matmul(pw,np.reshape([0,yu-yl,1],(3,1)))
    # print('c1',c1)
    # print('c3',c3)
    corners = np.int64(np.hstack([c1,c2,c3,c4]))
    # print(corners)
    x1 = min(corners[0,:])
    y1 = min(corners[1,:])
    x2 = max(corners[0,:])
    y2 = max(corners[1,:])
    # print(x1,y1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    # cv2.imshow('I',I)
    cv2.imwrite('./data/vase_modtry7/vm' + str(frno) + '.jpg',frame)
    cv2.imwrite('./data/vase_modtry7/warpedvm' + str(frno) + '.jpg',wIprev)
    print('frame',frno)

    # while True:
    #     if cv2.waitKey(00)==ord('q'):
    #         break
    if frno == 169:
        break
    pprev = pnew
    frno = frno + 1
    # temp = wIprev