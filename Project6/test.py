#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:41:10 2019

@author: kartikmadhira
"""

import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import skimage
import skimage.io
import tensorflow as tf


def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    conv1=tf.layers.conv2d(Img,filters=32,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm1 = tf.layers.batch_normalization(conv1)
    
    
    conv2=tf.layers.conv2d(norm1,
                           filters=32,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm2 = tf.layers.batch_normalization(conv2)
    
    pool2=tf.layers.max_pooling2d(inputs=norm2,
                                  pool_size=(2,2),
                                  strides=2)
    
    drop1=tf.layers.dropout(pool2,rate=0.5)

    conv3=tf.layers.conv2d(drop1,
                           filters=64,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm3 = tf.layers.batch_normalization(conv3)
    
    
    pool3=tf.layers.max_pooling2d(inputs=norm3,
                                  pool_size=(2,2),
                                  strides=2)
    
    drop2=tf.layers.dropout(pool3,rate=0.5)
    
    conv4=tf.layers.conv2d(drop2,
                           filters=64,
                           kernel_size=(5,5),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm4 = tf.layers.batch_normalization(conv4)

    pool4=tf.layers.max_pooling2d(inputs=norm4,
                                  pool_size=(2,2),
                                  strides=2)
    
    
    drop3=tf.layers.dropout(pool4,rate=0.5)
    
    
    conv5=tf.layers.conv2d(drop3,
                           filters=128,
                           kernel_size=(3,3),
                           padding="same",
                           activation=tf.nn.relu)
    
    norm5 = tf.layers.batch_normalization(conv5)
    
    
    pool5=tf.layers.max_pooling2d(inputs=norm5,
                                  pool_size=(2,2),
                                  strides=2)
    
    
    #now flattening the layer to add a fully connected layer
    flatLayer1=tf.reshape(pool5,[-1,pool5.shape[1:4].num_elements()])
    
    #adding a dense layer
    dense1=tf.layers.dense(inputs=flatLayer1,units=512,activation=tf.nn.relu)
    
    #add dropout if required!
    prLogits=tf.layers.dense(dense1,units=62,activation=None)
   # prLogits=tf.reshape(prLogits,[MiniBatchSize,62])
    prSoftMax=tf.nn.softmax(prLogits,name='softmax')
    
    
    return prLogits, prSoftMax


def inferSign(image):
    tf.reset_default_graph()
    ImgPH = tf.placeholder('float', shape=(1, 64, 64, 3))
    _, prSoftMaxS = CIFAR10Model(ImgPH, 0, 1)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess,'./data/Checkpoints/4model.ckpt')
        I1Batch=[]
        I1=image
        # I1=cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)
        #print(I1)
        I1 = (I1-np.mean(I1))/255
        
        I1=cv2.resize(I1,(64,64))
        I1Batch.append(I1)
        FeedDict = {ImgPH: I1Batch}
        PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
        predAcc=np.max(sess.run(prSoftMaxS, FeedDict))
        #print(np.max(sess.run(prSoftMaxS, FeedDict)))
        return PredT,predAcc
    
imageList=glob.glob('./data/input/*')
imageList.sort()
imageList=imageList[725:]


for img in imageList:
    
    firstImage=cv2.imread(img)
    
    #denoiseImage=cv2.fastNlMeansDenoisingColored(firstImage,None,10,10,7,21)
    #rgbImage=cv2.cvtColor(denoiseImage,cv2.COLOR_BGR2RGB)
    R=firstImage[:,:,2]
    G=firstImage[:,:,1]
    B=firstImage[:,:,0]
    R=np.float32(R)
    G=np.float32(G)
    B=np.float32(B)

    R=(R-np.min(R))*255/(np.max(R)-np.min(R))
    G=(G-np.min(G))*255/(np.max(G)-np.min(G))
    B=(B-np.min(B))*255/(np.max(B)-np.min(B))


    compareArrayBlue=np.subtract(B,R)/(R+G+B)
    compareArrayRed=np.minimum(np.subtract(R,B),np.subtract(R,G))/(R+G+B)

    cDashBlue=np.maximum(0,compareArrayBlue)
    cDashRed=np.maximum(0,compareArrayRed)

    #_,cDash=cv2.threshold(cDash,25,255,cv2.THRESH_BINARY)
    mask=np.ma.greater(cDashBlue,0.25)
    where_are_NaNs = np.isnan(cDashBlue)
    cDashBlue[where_are_NaNs] = 0
    #cDash=np.uint8(cDash)
    cDashBlue[mask]=255
    cDashBlue[500:,:]=0    
    cDashBlue=np.uint8(cDashBlue)
    
    mask=np.ma.greater(cDashRed,0.26)
    where_are_NaNs = np.isnan(cDashRed)
    cDashRed[where_are_NaNs] = 0
    #cDash=np.uint8(cDash)
    cDashRed[mask]=255
    cDashRed[500:,:]=0    
    cDashRed=np.uint8(cDashRed)
    
    #_,blueThreshold=cv2.threshold(blueRegion,150,255,cv2.THRESH_BINARY)
    mser = cv2.MSER_create(_delta=1,_min_diversity = 0.2,_min_area=600,_max_area=1500,_max_variation=0.15)
    grayBlue =cDashBlue
    grayRed=cDashRed
    vis = firstImage.copy()
    #detect regions in gray scale image
    regions, _ = mser.detectRegions(grayBlue)
    bbs=[]
    for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))

            if(h>=0.7*w):
                if(h<3*w):
                    buff=5
                    if(y-buff>10 and x-buff>10):
                        patch=vis[y-buff:y+h+buff,x-buff:x+w+buff,:]
                        #print(y-20,y+h+20,x-20,x+w+20)
                        patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                        pred,predAcc=inferSign(patch)
                    
                        #print(pred)
                        acc=0.90
                        if(pred==45 and predAcc>acc):
                            sign=skimage.io.imread('./data/Testing/00045/00080_00002.ppm')
                            sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                            if(x-40>10 and x+w-40>10 ):
                                vis[y-buff:y+h+buff,x-buff-40:x+w+buff-40,:]=sign
                        elif(pred==35 and predAcc>acc):
                            sign=skimage.io.imread('./data/Training/00035/00585_00000.ppm')
                            sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                            if(x-40>10 and x+w-40>10 ):
                                vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        elif(pred==38 and predAcc>acc):
                            sign=skimage.io.imread('./data/Training/00038/00452_00002.ppm')
                            sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                            if(x-40>10 and x+w-40>10 ):
                                vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        #plt.imshow(patch)
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,255,0),2)

    regions, _ = mser.detectRegions(grayRed)
    
    for i, region in enumerate(regions):
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        
        if(h>=0.7*w):
            if(h<1.5*w):
                buff=10
                if(y-buff>10 and x-buff>10):
                    patch=vis[y-buff:y+h+buff,x-buff:x+w+buff,:]
                    #print(y-20,y+h+20,x-20,x+w+20)
                    patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                    pred,predAcc=inferSign(patch)
                
                    #print(pred)
                    acc=0.85
                if(pred==21 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00021/00715_00000.ppm')
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
            #                             if(x-40>10 and x+w-40>10 ):
                    vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                elif(pred==14 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00014/00448_00001.ppm')
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
            #                             if(x-40>10 and x+w-40>10 ):
                    vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                elif(pred==1 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00001/00025_00000.ppm')
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
            #                             if(x-40>10 and x+w-40>10 ):
                    vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                elif(pred==17 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00017/00319_00002.ppm')
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    #                             if(x-40>10 and x+w-40>10 ):
                    vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                    #plt.imshow(patch)
                elif(pred==19 and predAcc>acc):
                        sign=skimage.io.imread(.'/data/Training/00019/00066_00002.ppm')
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        #                             if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,255,0),2)
    
    cv2.imshow('lol',vis)
    key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
    if key == 27:#if ESC is pressed,
        cv2.destroyAllWindows()
        break