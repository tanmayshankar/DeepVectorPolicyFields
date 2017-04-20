#!/usr/bin/env python 
from headers import *
import tensorflow as tf

class QMDP_RCNN():

	def __init__(self):

		self.discrete_x = 51
		self.discrete_y = 51
		self.discrete_z = 11

		self.input_x = 51
		self.input_y = 51
		self.input_z = 11

		self.conv1_size = 3
		self.conv2_size = 3
		self.conv3_size = 3

		self.conv1_stride = 1
		self.conv2_stride = 2
		self.conv3_stride = 2

		self.conv1_num_filters = 10
		self.conv2_num_filters = 10
		self.conv3_num_filters = 10

		# Remember, TensorFlow wants things as: Batch / Depth / Height / Width / Channels.
		# self.input = tf.placeholder(tf.float32,shape=[None,self.input_x,self.input_y,self.input_z],name='input')
		self.input = tf.placeholder(tf.float32,shape=[None,self.input_z,self.input_y,self.input_x],name='input')

		# self.reward = tf.placeholder(tf.float32,shape=[None,self.discrete_x,self.discrete_y,self.discrete_z],name='reward')
		# self.reward = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y,self.discrete_x],name='reward')

		# Defining convolutional layer 1:

		# Remember, depth, height, width, in channels, out channels.
		self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size,self.conv1_size,1,self.conv1_num_filters],stddev=0.1),name='W_conv1')
		self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[self.conv1_num_filters]),name='b_conv1')

		self.conv1 = tf.nn.conv3d(self.input,self.W_conv1,strides=[1,self.conv1_stride,self.conv1_stride,self.conv1_stride,1],padding='SAME') + self.b_conv1
		self.relu_conv1 = tf.nn.relu(self.conv1)

		# Defining convolutional layer 2: 
		
		# Conv layer 2: 
		self.W_conv2 = tf.Variable(tf.truncated_normal([self.conv2_size,self.conv2_size,self.conv2_size,self.conv1_num_filters,self.conv2_num_filters],stddev=0.1),name='W_conv2')
		self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[self.conv2_num_filters]),name='b_conv2')

		self.conv2 = tf.nn.conv3d(self.relu_conv1,self.W_conv2,strides=[1,self.conv2_stride,self.conv2_stride,self.conv2_stride,1],padding='SAME') + self.b_conv2
		self.relu_conv2 = tf.nn.relu(self.conv2)

		# Defining convolutional layer 3: 
		self.W_conv3 = tf.Variable(tf.truncated_normal([self.conv3_size,self.conv3_size,self.conv3_size,self.conv2_num_filters,self.conv3_num_filters],stddev=0.1),name='W_conv3')
		self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[self.conv3_num_filters]),name='b_conv3')

		# Reward is the "output of this convolutional layer.	"
		self.reward = tf.nn.conv3d(self.relu_conv2,self.W_conv3,strides=[1,self.conv3_stride,self.conv3_stride,self.conv3_stride,1],padding='SAME') + self.b_conv3

	





