#!/usr/bin/env python

# Loading header files.
import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import tensorflow_fcn.fcn16_vgg as fcn16_vgg
import utils
import matplotlib.pyplot as plt
import copy
from tensorflow.python.framework import ops

# Environment configuration.
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)


# Telling Tensorflow what GPUs to use.
gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="1,2")
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

# Creating an image placeholder.
images = tf.placeholder("float")
# Initializing the network.
vgg_fcn = fcn16_vgg.FCN16VGG()

# Just some random crap to initiailize the net with.
i=0
# img1 = skimage.io.imread("image_{0}.png".format(i))      
x = str(i)
x = x.rjust(5,'0')

FILE_DIR = "/home/tanmay/Research/DeepVectorPolicyFields/Data/Cars/pdtv_frames/"
img1 = skimage.io.imread(FILE_DIR+"Sequence_0/{0}.png".format(x))      

feed_dict = {images: img1}
batch_images = tf.expand_dims(images, 0)

with tf.name_scope("content_vgg"):
	vgg_fcn.build(batch_images, debug=True)

print('Finished building Network.')

init = tf.initialize_all_variables()
sess.run(tf.initialize_all_variables())

# Setting the class we care about in life: CARS



k=11
factor = 5
seqfl = npy.load(FILE_DIR+"File_list.npy")
# NOW ACTUALLY GOING TO RUN THIS NETWORK OVER ALL THE IMAGES:
for seq in range(15):
	for i in range(len(seqfl)/factor):

		# Logistic stuff
		x = str(factor*i)
		x = x.rjust(5,'0')

		# Load the image.
		print("Running on Sequence",seq,"Image:",i)		
		img1 = skimage.io.imread(FILE_DIR+"Sequence_{0}/{1}.jpg".format(seq,x))
		
		# Feed the image.
		feed_dict = {images: img1}
		batch_images = tf.expand_dims(images, 0)

		tensors = [vgg_fcn.pred, vgg_fcn.pred_up, vgg_fcn.upscore32]
		down, up, score = sess.run(tensors, feed_dict=feed_dict)
		
		score_reshape = score.reshape((20,img1.shape[0],img1.shape[1]))                  

		# Building the Heatmap

		belief = np.multiply(up[0]==k,score_reshape[k])
			# belief = (up[0]==k)*score_reshape[k]*

		for a in range(0,3):
			img[:,:,a] = (up[0]==k)*img1[:,:,a]
			# img = np.multiply(up[0]==k,img1)   
		belief += belief.min()    
		belief /= belief.sum()

		# Saving files.
		npy.save(FILE_DIR+"Sequence_{0}/class_{1}.npy".format(seq,x),up[0])
		npy.save(FILE_DIR+"Sequence_{0}/scores_{1}.npy".format(seq,x),score_reshape)
		npy.save(FILE_DIR+"Sequence_{0}/belief_{1}.npy".format(seq,x),belief)
