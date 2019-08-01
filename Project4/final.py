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

    for i in range(t.shape[1]):            # x
        for j in range(t.shape[0]):       # y
            x = np.array([i,j,1]).T
            x = np.reshape(x,(3,1))
            W = np.int16(np.round(np.matmul(pw,x)))
            a = I[W[1],W[0]]
            igx = Igradx[W[1],W[0]]
            igy = Igrady[W[1],W[0]]
            wI[j,i] = a
            wIgradx[j,i] = igx
            wIgrady[j,i] = igy
    r = np.median(t)/np.median(wI)
    wI = np.uint8(r*wI)
    flag = False
    if r<0.5:
        flag = True
    return wI, wIgradx, wIgrady,flag


def computeP(temp,I,pprev,thresh,Igradx,Igrady):
    p = pprev
    delpnorm = thresh + 10
    it = np.int(0)
    delpnormprev = 10
    while (delpnorm > thresh):
        wI, wIgradx, wIgrady,flag = warpAll(temp,I,p,Igradx,Igrady)
        # plt.imshow(wI)
        # plt.show()
        error = temp - wI
        error = np.reshape(error,(error.shape[0]*error.shape[1],1))
        sigma = np.std(error)
        print(sigma)
        s1 = np.zeros([6,1])
        s2 = np.zeros([6,6])
        for i in range(temp.shape[1]):            # x
            for j in range(temp.shape[0]):       # y
                wJ = np.array([[i,0,j,0,1,0],[0,i,0,j,0,1]])
                ig = np.array([wIgradx[j,i],wIgrady[j,i]])
                b1 = np.matmul(ig,wJ)
                err = temp[j,i] - wI[j,i]
                t = err**2
                if t<=sigma**2:
                    rho = 0.5*t
                elif t>sigma**2:
                    rho = sigma*np.sqrt(t) - 0.5*sigma**2
                else :
                    rho = 1
                b1 = np.reshape(b1,(1,6))
                err = np.reshape(err,(1,1))
                s1 = s1 + rho*b1.T*err
                s2 = s2 + rho*b1.T*b1

        H = s2
        sdpu = s1

        delp = np.matmul(np.linalg.inv(H),sdpu)

        # pnew = delp
        # pinv = np.array([-pnew[0] - pnew[0]*pnew[3] + pnew[1]*pnew[2]
        #             ,-pnew[1]
        #             ,-pnew[2]
        #             ,-pnew[3] - pnew[0]*pnew[3] + pnew[1]*pnew[2]
        #             ,-pnew[4] - pnew[3]*pnew[4] + pnew[2]*pnew[5]
        #             ,-pnew[5] - pnew[0]*pnew[5] + pnew[1]*pnew[4]])
        # pinv = (1/((1+pnew[0])*(1+pnew[3]) - pnew[1]*pnew[2]))*pinv
        #
        # print("pinv",pinv.shape)
        # print("p",p.shape)
        # p = np.multiply(p,pinv)
        p= p+delp
        # p[4] = p[4]
        # p[5] = p[5]
        # p = np.array([p[0]+pinv[0]+p[0]*pinv[0]+p[2]*p[1],
        #               p[1]+pinv[1]+p[1]*pinv[0]+p[3]*p[2],
        #               p[2]+pinv[2]+p[0]*pinv[2]+p[2]*p[3],
        #               p[3]+pinv[3]+p[1]*pinv[2]+p[3]*p[3],
        #               p[4]+pinv[4]+p[0]*pinv[4]+p[2]*p[5],
        #               p[5]+pinv[5]+p[1]*pinv[4]+p[3]*p[5],])
        # print("new",p.shape)
        # print("delta P",delp.shape)
        delpnorm = np.linalg.norm(delp)

        if delpnormprev > delpnorm:
            pmin = p
        delpnormprev = delpnorm
        print('dpnorm',delpnorm)

        if  (it > 50):
            p = pmin
            break
        it = it + 1

    return p,wI,flag

img1 = cv2.imread('./data/vase/0019.jpg')
# temp = np.zeros([60,80])
xl = 125
xu = 170
yl = 92
yu = 148
temp = img1[yl:yu,xl:xu]
temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
cv2.imwrite('./data/vase_template.jpg',temp)
p = np.zeros([6,1])
p[4] = xl
p[5] = yl
pw = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]])
pw = np.reshape(pw,(2,3))
c1 = np.matmul(pw,np.reshape([xl,yl,1],(3,1)))
c2 = np.matmul(pw,np.reshape([xu,yl,1],(3,1)))
c3 = np.matmul(pw,np.reshape([xu,yu,1],(3,1)))
c4 = np.matmul(pw,np.reshape([xl,yu,1],(3,1)))
print('c1',c1)
print('c3',c3)
print(temp.shape)

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
    thresh = 0.01
    r = np.mean(temp)/np.mean(I)
    I = np.uint8(r*I)
    pnew, wIprev,flag = computeP(temp,I,pprev,thresh,Igradx,Igrady)
    pprev = pnew
    # print("threshold",r)
    if flag ==True:
        print("#####################################")
        temp = wIprev
    # print('pnew',pnew)
    pw = np.array([[1+pnew[0], pnew[2], pnew[4]],[pnew[1], 1+pnew[3], pnew[5]]])
    pw = np.reshape(pw,(2,3))
    c1 = np.matmul(pw,np.reshape([0,0,1],(3,1)))
    c2 = np.matmul(pw,np.reshape([xu-xl,0,1],(3,1)))
    c3 = np.matmul(pw,np.reshape([xu-xl,yu-yl,1],(3,1)))
    c4 = np.matmul(pw,np.reshape([0,yu-yl,1],(3,1)))
    corners = np.int64(np.hstack([c1,c2,c3,c4]))
    print(corners)
    x1 = min(corners[0,:])
    y1 = min(corners[1,:])
    x2 = max(corners[0,:])
    y2 = max(corners[1,:])
    print(x1,y1)

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.imwrite('./data/vase_modtry7/vm' + str(frno) + '.jpg',frame)

    if frno == 280:
        break
    pprev = pnew
    frno = frno + 1