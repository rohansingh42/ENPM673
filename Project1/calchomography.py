import argparse
import os, sys
import pickle
import numpy as np

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import imutils

import matplotlib.pyplot as plt
