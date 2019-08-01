import numpy as np

import cv2
import matplotlib.pyplot as plt
import glob
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


filenames = [img for img in glob.glob('./Oxford_dataset/stereo/centre/*png')]

filenames.sort() # ADD THIS LINE

i=1
for img in filenames:
    # print(img)
    im = np.array(Image.open(img))
    # plt.imshow(im)
    # print(im.shape)
    image = cv2.cvtColor(im,cv2.COLOR_BayerGR2BGR)
    cv2.imwrite('./Oxford_dataset/mod_frames/frame' + str(i) + '.jpg',image)
    i = i + 1
    # image_list.append(im)
    # print(img)

