#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform
from skimage import measure

import os
import scipy as scp
import scipy.misc

import numpy as npy
import sys
import matplotlib.pyplot as plt
import copy

FILE_DIR = "/home/tanmay/Research/DeepVectorPolicyFields/Data/Cars/pdtv_frames/"
roadmap = npy.loadtxt(FILE_DIR+'ROADMAP.txt')

seqfl = npy.load(FILE_DIR+"File_list.npy")
threshold = 100
factor = 5


for i in range(15):
	for j in range(len(seqfl[i])/factor):
		x = str(j)
		x = x.rjust(5,'0')
		print("Processing Sequence: ",i,"Image:",j)

		classes = npy.load(FILE_DIR+"Sequence_{0}/class_{1}.npy".format(i,x))
		belief_maps = npy.zeros((10,480,640))

		# for k in range(2):
		# 	mask = classes[k]*roadmap
		# 	# Running a labelling algorithm
		# 	labels = measure.label(mask)
		# 	# Range over number of instances of a particular class.
		# 	label_list = npy.zeros((labels.max()),dtype=int)
		# 	# For all the instances, count number of pixels that exist for each label.
		# 	for sr in range(labels.max()):
		# 		label_list[sr] = ((labels==sr)*npy.ones((480,640))).sum()
		# 	# Sort.
		# 	sort = npy.argsort(-label_list)

		# 	for srt in range(1,min(11))
		
		classes = npy.sum(classes,axis=0)
		mask = classes*roadmap
		labels = measure.label(mask)

		label_list = npy.zeros(labels.max()).astype(int)	
		for ji in range(labels.max()):
			label_list[ji] = ((labels==ji)*npy.ones((480,640))).sum()
		sort = npy.argsort(-label_list)
		
		for k in range(1,min(11,sort.shape[0])):		
			if (label_list[sort[k]]>threshold):
				belief_maps[k-1] = (labels==sort[k])*npy.ones((480,640))		

		npy.save(FILE_DIR+"Sequence_{0}/belief_map_{1}.npy".format(i,x),belief_maps)

# for i in range(0,448):	
# 	print("Running on Image",i)
# 	belief_maps = npy.zeros((10,480,640))
# 	classes = npy.loadtxt("class_{0}.txt".format(i))
# 	mask = (classes==11)*roadmap
# 	labels = measure.label(mask)

# 	label_list = npy.zeros(labels.max()).astype(int)	
# 	for j in range(labels.max()):
# 		label_list[j] = ((labels==j)*npy.ones((480,640))).sum()
# 	sort = npy.argsort(-label_list)
	
# 	for k in range(1,min(11,sort.shape[0])):		
# 		if (label_list[sort[k]]>threshold):
# 			belief_maps[k-1] = (labels==sort[k])*npy.ones((480,640))		

# 	npy.save(FILE_DIR+"Sequence_{0}/belief_map_{1}.npy".format(i,x),belief_maps)
# 	with file('belief_maps_{0}.txt'.format(i),'w') as outfile:
# 		for data in belief_maps:
# 			outfile.write('#Belief_Map.\n')
# 			npy.savetxt(outfile,data,fmt='%i')