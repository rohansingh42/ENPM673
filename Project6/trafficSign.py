
import os, sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import skimage
import skimage.io
import tensorflow as tf


# def apnaModel(Img, ImageSize, MiniBatchSize):
#     """
#     Inputs:
#     Img is a MiniBatch of the current image
#     ImageSize - Size of the Image
#     Outputs:
#     prLogits - logits output of the network
#     prSoftMax - softmax output of the network
#     """
#     #Define Filter parameters for the first convolution layer
#     # filter_size1 = 5
#     # num_filters1 = 64

#     #Define Filter parameters for the second convolution layer
#     # filter_size2 = 5

#     #Define number of neurons in hidden layer
#     # fc_size1 = 256
#     # fc_size2 = 128
#     # num_filters2 = 64


#     #Define number of class labels
#     num_classes = 62
#     #############################
#     # Fill your network here!
#     #############################


#     #Construct first convolution layer
#     net = Img
#     net = tf.layers.conv2d(inputs = net, name='layer_conv1', padding='same',filters = 32, kernel_size = 5, activation = None)
#     net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn1')
#     net = tf.nn.relu(net, name = 'layer_Relu1')
#     # layer_conv1 = net
#     net = tf.layers.conv2d(inputs = net, name = 'layer_conv2', padding= 'same', filters = 32, kernel_size = 5, activation = None)
#     net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2')
#     net = tf.nn.relu(net, name = 'layer_Relu2')
#     # layer_conv2 = net
#     net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

#     net = tf.layers.dropout(net,rate=0.5)

#     net = tf.layers.conv2d(inputs = net, name = 'layer_conv3', padding= 'same', filters = 64, kernel_size = 5, activation = None)
#     net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn3')
#     net = tf.nn.relu(net, name = 'layer_Relu3')

#     net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

#     net = tf.layers.conv2d(inputs = net, name = 'layer_conv4', padding= 'same', filters = 32, kernel_size = 5, activation = None)
#     net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn4')
#     net = tf.nn.relu(net, name = 'layer_Relu4')

#     net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

#     net = tf.layers.flatten(net)

#     #Define the Neural Network's fully connected layers:
#     net1 = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

#     # net2 = tf.layers.dense(inputs = net1, name ='layer_fc2',units=256, activation=tf.nn.relu)

#     # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=128, activation=tf.nn.relu)

#     net3 = tf.layers.dense(inputs = net1, name='layer_fc_out', units = num_classes, activation = None)

#     #prLogits is defined as the final output of the neural network
#     # prLogits = layer_fc2
#     prLogits = net3
#     #prSoftMax is defined as normalized probabilities of the output of the neural network
#     prSoftMax = tf.nn.softmax(logits = prLogits)

#     return prLogits, prSoftMax

def apnaModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer
    # filter_size1 = 5
    # num_filters1 = 64

    #Define Filter parameters for the second convolution layer
    # filter_size2 = 5

    #Define number of neurons in hidden layer
    # fc_size1 = 256
    # fc_size2 = 128
    # num_filters2 = 64


    #Define number of class labels
    num_classes = 62
    #############################
    # Fill your network here!
    #############################


    #Construct first convolution layer
    net = Img
    net = tf.layers.conv2d(inputs = net, name='layer_conv1', padding='same',filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn1')
    net = tf.nn.relu(net, name = 'layer_Relu1')
    # layer_conv1 = net
    net = tf.layers.conv2d(inputs = net, name = 'layer_conv2', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2')
    net = tf.nn.relu(net, name = 'layer_Relu2')
    # layer_conv2 = net
    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = tf.layers.dropout(net,rate=0.5)

    net = tf.layers.conv2d(inputs = net, name = 'layer_conv3', padding= 'same', filters = 64, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn3')
    net = tf.nn.relu(net, name = 'layer_Relu3')


    net = tf.layers.conv2d(inputs = net, name = 'layer_conv4', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn4')
    net = tf.nn.relu(net, name = 'layer_Relu4')

    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net1 = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    net2 = tf.layers.dense(inputs = net1, name ='layer_fc2',units=256, activation=tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=128, activation=tf.nn.relu)

    net3 = tf.layers.dense(inputs = net2, name='layer_fc_out', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net3
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax


def inferSign(image):
    tf.reset_default_graph()
    ImgPH = tf.placeholder('float', shape=(1, 64, 64, 3))
    _, prSoftMaxS = apnaModel(ImgPH, 0, 1)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        # Saver.restore(sess,'./Checkpoints/19model.ckpt')
        Saver.restore(sess,'/home/rohan/CMSC-733/Abhi1625_hw0/Phase2/Checkpoints/4model.ckpt')
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
frno = 2489
imageList=imageList[frno:]



for img in imageList:
    # print(img)
    print(frno)
    firstImage=cv2.imread(img)
    
    rChannel=firstImage[:,:,2]
    gChannel=firstImage[:,:,1]
    bChannel=firstImage[:,:,0]
    rChannel=np.float32(rChannel)
    gChannel=np.float32(gChannel)
    bChannel=np.float32(bChannel)

    rChannel=(rChannel-np.min(rChannel))*255/(np.max(rChannel)-np.min(rChannel))
    gChannel=(gChannel-np.min(gChannel))*255/(np.max(gChannel)-np.min(gChannel))
    bChannel=(bChannel-np.min(bChannel))*255/(np.max(bChannel)-np.min(bChannel))


    compareArrayBlue=np.subtract(bChannel,rChannel)/(rChannel+gChannel+bChannel)
    compareArrayRed=np.minimum(np.subtract(rChannel,bChannel),np.subtract(rChannel,gChannel))/(rChannel+gChannel+bChannel)

    cDashBlue = np.maximum(0,compareArrayBlue)
    cDashRed = np.maximum(0,compareArrayRed)


    mask=np.ma.greater(cDashBlue,0.25)
    pos_NaNs = np.isnan(cDashBlue)
    cDashBlue[pos_NaNs] = 0
    cDashBlue[mask]=255
    cDashBlue[500:,:]=0    
    cDashBlue=np.uint8(cDashBlue)
    
    mask=np.ma.greater(cDashRed,0.26)
    pos_NaNs = np.isnan(cDashRed)
    cDashRed[pos_NaNs] = 0
    #cDash=np.uint8(cDash)
    cDashRed[mask]=255
    cDashRed[500:,:]=0    
    cDashRed=np.uint8(cDashRed)
    
    #_,blueThreshold=cv2.threshold(blueRegion,150,255,cv2.THRESH_BINARY)
    mser = cv2.MSER_create(_delta=10,_min_diversity = 0.2,_min_area=400,_max_area=3000,_max_variation=0.35)
    # bmser = cv2.MSER_create(10, 100, 1000, 0.5, 0.2, 200, 1.01, 0.003, 5)
    # rmser = cv2.MSER_create(10, 100, 1000, 0.5, 0.2, 200, 1.01, 0.003, 5)
    grayBlue =cDashBlue
    grayRed=cDashRed
    vis = firstImage.copy()
    #detect regions in gray scale image
    regions, _ = mser.detectRegions(grayBlue)
    bbs=[]
    for i, region in enumerate(regions):
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))

        if(h>=0.65*w):
            if(h<2.5*w):
                buff=5
                if(y-buff>5 and x-buff>5):
                    patch=vis[y-buff:y+h+buff,x-buff:x+w+buff,:]
                    #print(y-20,y+h+20,x-20,x+w+20)
                    patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                    pred,predAcc=inferSign(patch)
                
                    print(pred)
                    print(predAcc)
                    acc=0.85
                    if(pred==45 and predAcc>acc):
                        sign=skimage.io.imread('./data/Training/00045/00318_00001.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        # if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff-40:x+w+buff-40,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        print('box')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)


                    elif(pred==35 and predAcc>acc):
                        sign=skimage.io.imread('./data/Training/00035/00585_00000.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        # if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                    elif(pred==38 and predAcc>acc):
                        sign=skimage.io.imread('./data/Training/00038/00452_00002.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        # if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                    #plt.imshow(patch)
                        
    regions, _ = mser.detectRegions(grayRed)
    
    for i, region in enumerate(regions):
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        
        if(h>=0.8*w):
            if(h<2.5*w):
                buff=10
                if(y-buff>10 and x-buff>10):
                    patch=vis[y-buff:y+h+buff,x-buff:x+w+buff,:]
                    #print(y-20,y+h+20,x-20,x+w+20)
                    patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                    pred,predAcc=inferSign(patch)
                
                    print(pred)
                    print(predAcc)
                    acc=0.8
                if(pred==21 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00021/00715_00000.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==14 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00014/00448_00001.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==1 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00001/00025_00000.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==17 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00017/00319_00002.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')
                        #plt.imshow(patch)

                elif(pred==19 and predAcc>acc):
                    sign=skimage.io.imread('./data/Training/00019/00066_00002.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        vis[y-buff:y+h+buff,x-buff:x+w+buff,:]=sign
                        cv2.rectangle(vis,(x-buff,y-buff),(x+w+buff,y+h+buff),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(vis,str(pred),(x+buff,y+buff), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

    
    vis = cv2.resize(vis,(480,320))
    cv2.imwrite('./results/det_and_class/frame' + str(frno) + '.png',vis)
    # cv2.imshow('lol',vis)
    # key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
    # if key == 27:#if ESC is pressed,
    #     cv2.destroyAllWindows()
    #     break
    frno = frno + 1
