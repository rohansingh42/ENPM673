import os, sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import numpy as np
import cv2
import os
import copy
from PIL import Image
from skimage.feature import hog
from skimage import feature, exposure
from sklearn import svm
# from scipy.misc import imread

path = "./data/input/"

train_set = [["./data/Training/00001", 1], ["./data/Training/00014", 14], ["./data/Training/00017", 17], ["./data/Training/00019", 19], 
                ["./data/Training/00021", 21], ["./data/Training/00035", 35], ["./data/Training/00038", 38], ["./data/Training/00045", 45]]

frames = []
for frame in os.listdir(path):
    frames.append(frame)
    frames.sort()

###########################################################################################
#####################################Training##############################################
hog_list = []
label_list = []
count = 0

for name in train_set:
    value = name[0]
    label = name[1]
    image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.ppm')]
    #print('number of images in the folder for ',label,"-",len(image_list))
    for image in image_list:
        count += 1
        im = np.array(Image.open(image))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_prep = cv2.resize(im_gray, (64, 64))
        
        fd, h = feature.hog(im_prep, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            # transform_sqrt=True, block_norm="L1", 
                            visualise=True)
        hog_list.append(h)
        label_list.append(label)
        
list_hogs = []
for hogs in hog_list:
    hogs = hogs.reshape(64*64)
    list_hogs.append(hogs)

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(list_hogs, label_list)
###########################################################################################


#%%
def Classify(image):
    hog_list_test = []
    fd, h = feature.hog(image, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            # transform_sqrt=True, block_norm="L1", 
                            visualise=True)
    hog_list_test.append(h)
    list_hogs_test = []
    for hogs in hog_list_test:
        hogs = hogs.reshape(64*64)
        list_hogs_test.append(hogs)
    
    predictions = []
    predictions = clf.predict(list_hogs_test)
    print(predictions[0])

    if predictions[0] in [1, 17, 14, 19, 21, 35, 38, 45]:
        result = cv2.imread('./results/detection_Sayan/'+str(predictions[0])+'.PNG')
    
    return result
    
def findBoundingBox(contours):
    centerdict = {}
    
    for i in range(0, len(contours)):
        M = cv2.moments(contours[i])
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if i == 0:
                centerdict[(cX, cY)] = [i]
            else:
                flag = 0
                for key in list(centerdict.keys()):
                    if (cX - key[0])**2 + (cY - key[1])**2 - 100**2 < 0:
                        centerdict[key].append(i)
                        flag = 1
                        break
                if flag == 0:
                    centerdict[(cX, cY)] = [i]
    
    boxes = [] 
    for key in list(centerdict.keys()):
        temp = 0
        for index in centerdict[key]:
            area = cv2.contourArea(contours[index])
            if area > temp:
                temp = area
                main = contours[index]
        boxes.append(main)
            
    #print(centerdict)
    #print(len(boxes))
    return boxes
                                       
for index in range(1050, len(frames)):
    #print(index)
    img = cv2.imread("./data/input/" + str(frames[index])) # index # 100
    img1 = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(img1, None,10,10,7,21)

    bluec = dst[:,:,0]
    greenc = dst[:,:,1]
    redc = dst[:,:,2]

    minmax_img_b = bluec - np.min(bluec)
    minmax_img_b = minmax_img_b/(np.max(bluec)-np.min(bluec))
    minmax_img_b = minmax_img_b * 255

    minmax_img_g = greenc - np.min(greenc)
    minmax_img_g = minmax_img_g/(np.max(greenc)-np.min(greenc))
    minmax_img_g = minmax_img_g * 255

    minmax_img_r = redc - np.min(redc)
    minmax_img_r = minmax_img_r/(np.max(redc)-np.min(redc))
    minmax_img_r = minmax_img_r * 255

    zero_img = np.zeros((img1.shape[0],img1.shape[1]))

    num = minmax_img_b - minmax_img_r
    den = minmax_img_b + minmax_img_g + minmax_img_r
    total = num/den
    total = np.where(np.invert(np.isnan(total)), total, 0)

    normalize_img_b = (np.maximum(zero_img, total)*255).astype(np.uint8)

    num1 = minmax_img_r - minmax_img_b
    num2 = minmax_img_r - minmax_img_g
    den1 = minmax_img_b + minmax_img_g + minmax_img_r
    total1 = np.minimum(num1, num2)/den1
    total1 = np.where(np.invert(np.isnan(total1)), total1, 0)

    normalize_img_r = (np.maximum(zero_img, total1)*255).astype(np.uint8)
    normalize_img_b=np.where(normalize_img_b < 35, normalize_img_b, 255) 
    #print(normalize_img_b)
    #cv2.imshow("1",normalize_img_b)
    normalize_img_r=np.where(normalize_img_r < 20, normalize_img_r, 255) 
    #print(normalize_img_b)
    #cv2.imshow("2",normalize_img_r)
    #mser = cv2.MSER_create(5,100,14400,0.25,0.2,200,1.01,0.003,5)
    bmser = cv2.MSER_create(10, 100, 1000, 0.5, 0.2, 200, 1.01, 0.003, 5)
    rmser = cv2.MSER_create(10, 100, 1000, 5, 0.2, 200, 1.01, 0.003, 5)
    
    imagecopy = img1.copy()
    regions, _ = bmser.detectRegions(normalize_img_b)
    blueregions =  findBoundingBox(regions)
    
    for region in blueregions:
        x,y,w,h = cv2.boundingRect(region)
        #print(h/w)
        if  y < 200 and (w/h) < 1.1 :# and (h/w) >= 0.5:
            cv2.rectangle(imagecopy,(x,y),(x+w,y+h),(0,255,0),2)
            testimage = imagecopy[y:y+h, x:x+w]
            testimagegray = cv2.cvtColor(testimage, cv2.COLOR_BGR2GRAY)
            testimageresized = cv2.resize(testimagegray, (64, 64))
            #cv2.imshow("1",testimageresized)
            resultimage = Classify(testimageresized)
            #cv2.imshow("2",resultimage)
            resultimageresized = cv2.resize(resultimage, (w, h))
            #print(resultimageresized.shape)
            if x-w > 0:
                imagecopy[y:y+h, x-w:x] = resultimageresized
            else:
                imagecopy[y:y+h, x+w:x+2*w] = resultimageresized
    
    regions1, _ = rmser.detectRegions(normalize_img_r)
    #print('red')
    redregions =  findBoundingBox(regions1)
    
    for region1 in redregions:
        area = cv2.contourArea(region1)
        x1,y1,w1,h1 = cv2.boundingRect(region1)
        if y1 < 90 and x1> 400  and (w1/h1) < 1.1 and area > 170 :
            print(area)
            cv2.rectangle(imagecopy,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
            testimager = imagecopy[y1:y1+h1, x1:x1+w1]
            testimagegrayr = cv2.cvtColor(testimager, cv2.COLOR_BGR2GRAY)
            testimageresizedr = cv2.resize(testimagegrayr, (64, 64))
            resultimager = Classify(testimageresizedr)
            print(resultimager)
            resultimageresizedr = cv2.resize(resultimager, (w1, h1))
            if x1-w1 > 0:
                imagecopy[y1:y1+h1, x1-w1:x1] = resultimageresizedr
            else:
                imagecopy[y1:y1+h1, x1+w1:x1+2*w1] = resultimageresizedr
            
    #cv2.imshow('normalize1', normalize_img_b)
    #cv2.imshow('normalize2', normalize_img_r)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imagecopy,str(index),(10,500), font, 4,(0,0,0),2,cv2.LINE_AA)
    
    cv2.imshow('main', imagecopy)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()


#%%



