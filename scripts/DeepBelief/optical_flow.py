#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform
import skimage.color as sic 

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import matplotlib.pyplot as plt
import copy
import cv2

i=0
img0 = skimage.io.imread("image_{0}.png".format(i))  
img0g = sic.rgb2gray(img0)

for i in range(0,447):

    print("Running on Image", i)
    img1 = skimage.io.imread("image_{0}.png".format(i+1))
	img1g = sic.rgb2gray(img1)
	flow = cv2.calcOpticalFlowFarneback(img0g,img1g,None,0.5,3,15,3,5,1.2,0)
	img0g = copy.deepcopy(img1g)

	with file('flow_{0}.txt'.format(i),'w') as outfile:
		for data in flow:
			outfile.write('#Flow_Values.\n')
			np.savetxt(outfile,100000*data,fmt='%-7.5f')
