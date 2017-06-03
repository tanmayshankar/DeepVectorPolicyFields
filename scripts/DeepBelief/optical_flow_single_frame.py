#!/usr/bin/env python
import skimage
import skimage.io
import skimage.transform
import skimage.color as sic 

import os
import scipy as scp
import scipy.misc

import numpy as npy
import logging
import tensorflow as tf
import sys

import matplotlib.pyplot as plt
import copy
import cv2

i=0
seqfl = npy.load("File_list.npy")
factor = 5

for i in range(15):
	# for j in range(len(seqfl[i])):
	for j in range(len(seqfl[i])/factor):

		print("Running on sequence",i,"image:",j)
		x = str(factor*j)
		x = x.rjust(5,'0')
		y = str(factor*j+1)
		y = y.rjust(5,'0')
		img1 = skimage.io.imread("Sequence_{0}/{1}.jpg".format(i,x))
		img1_gray = sic.rgb2gray(img1)

		img2 = skimage.io.imread("Sequence_{0}/{1}.jpg".format(i,y))
		img2_gray = sic.rgb2gray(img2)

		flow = cv2.calcOpticalFlowFarneback(img1_gray,img2_gray,None,0.5,3,15,3,5,1.2,0)
		npy.save("Sequence_{0}/flow_{1}.npy".format(i,j),flow)


# for i in range(0,448):
# 	print("Running on Image", i*10)
# 	img0 = skimage.io.imread("image_{0}.png".format(i*10))  
# 	img0g = sic.rgb2gray(img0)    
#     img1 = skimage.io.imread("image_{0}.png".format(i*10+1))
# 	img1g = sic.rgb2gray(img1)
# 	flow = cv2.calcOpticalFlowFarneback(img0g,img1g,None,0.5,3,15,3,5,1.2,0)
# 	img0g = copy.deepcopy(img1g)

# 	with file('flow_{0}.txt'.format(i),'w') as outfile:
# 		for data in flow:
# 			outfile.write('#Flow_Values.\n')
# 			npy.savetxt(outfile,100000*data,fmt='%-7.5f')
