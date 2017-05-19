#!/usr/bin/env python 
from headers import *
import tensorflow as tf

class QMDP_RCNN():

	def __init__(self):

		# Defining common variables.
		self.discrete_x = 33
		self.discrete_y = 33
		self.discrete_z = 22	
		self.action_size = 6

		# angular discretization.
		self.discrete_theta = 18
		# self.dummy_zeroes = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z, self.action_size))
		self.dummy_zeroes = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z, self.discrete_theta, self.action_size))

		self.dimensions = 3
		self.angular_dimensions = 2
		self.action_size = 6

		# Final number of convolutional filters must be equal to the action space x discrete_theta.
		# self.conv3_num_filters = self.action_size
		self.conv3_num_filters = self.action_size*self.discrete_theta

		# Setting discretization variables
		self.action_lower = -1
		self.action_upper = +1
		self.action_cell = npy.ones(self.dimensions)

		# Assuming lower is along all dimensions. 		
		self.traj_lower = npy.array([-1,-1,0])
		self.traj_upper = +1

		self.ang_traj_lower = -1
		self.ang_traj_upper = +1

		# ALWAYS EXPRESSING PHI BEFORE Theta in Indexing
		self.grid_cell_size = npy.array((self.traj_upper-self.traj_lower)).astype(float)/[self.discrete_x-1, self.discrete_y-1, self.discrete_z-1]		
		self.angular_grid_cell_size = npy.array((self.ang_traj_upper-self.ang_traj_lower)).astype(float)/[self.discrete_theta]				

		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		self.angular_action_size = 2
		self.angular_action_space = npy.array([[-1,0],[1,0]])
		# Negative theta, positive theta.

		# Defining transition model.
		self.trans_space = 3
		
		self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		# Defining angular transition models. 
		self.angular_trans = npy.ones((self.angular_action_size,self.trans_space))
		
		for k in range(self.action_size):
			self.trans[k] /= self.trans[k].sum()

		# Normalizing angular transition models.
		for k in range(self.angular_action_size):			
			self.angular_trans[k] /= self.angular_trans[k].sum()

		self.action_counter = npy.zeros(self.action_size+self.angular_action_size)
		# Defining observation model.
		self.obs_space = 5
		
		# Defining belief variables.
		self.from_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.to_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.target_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.intermed_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.sensitivity = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))

		# Defining angular beliefs.
		self.from_angular_belief = npy.zeros((self.discrete_theta))
		self.to_angular_belief = npy.zeros((self.discrete_theta))
		self.target_angular_belief = npy.zeros((self.discrete_theta))
		self.intermed_angular_belief = npy.zeros((self.discrete_theta))
		self.sensitivity_angular_belief = npy.zeros((self.discrete_theta))

		# Defining extended belief states. 
		self.w = self.trans_space/2
		self.to_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))
		self.from_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))

		# Extended angular belief states. 
		self.to_angular_ext = npy.zeros((self.discrete_theta+2*self.w))
		self.from_angular_ext = npy.zeros((self.discrete_theta+2*self.w))

		# Defining trajectory
		self.orig_traj = []
		self.orig_vel = []

		self.beta = npy.zeros(self.action_size)
		self.angular_beta = npy.zeros(self.angular_action_size)

		# Defining observation model related variables. 
		self.obs_space = 4
		self.obs_model = npy.zeros((self.obs_space,self.obs_space,self.obs_space))
		self.angular_obs_model = npy.zeros((self.obs_space))

		self.h = self.obs_space-1
		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))
		self.extended_angular_obs_belief = npy.zeros((self.discrete_theta+2*self.h))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 2
		self.annealing_rate = 0.1

		# Setting training parameters: 
		self.epochs = 20


		# Defining Value function, policy, etc. 	
		self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z, self.discrete_theta))
		self.policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_theta))
		self.reward = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_theta))
		self.Qvalues = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_theta))

		# Defining belief space Q values; softmax values. 
		self.belief_space_q = npy.zeros(self.action_size)
		self.softmax_q = npy.zeros(self.action_size)

		# Discount
		self.gamma = 0.95


	# def initialize_tensorflow_model(self,sess):

	# 	# Initializing Tensorflow Session: 
	# 	self.sess = sess

	# 	# Remember, TensorFlow wants things as: Batch / Depth / Height / Width / Channels.
	# 	# self.input = tf.placeholder(tf.float32,shape=[None,self.input_z,self.input_y,self.input_x,1],name='input')
	# 	# Introducing RGB channel: 
	# 	self.input = tf.placeholder(tf.float32,shape=[None,self.input_z,self.input_y,self.input_x,3],name='input')

	# 	# self.reward = tf.placeholder(tf.float32,shape=[None,self.discrete_x,self.discrete_y,self.discrete_z],name='reward')

	# 	# DEFINING CONVOLUTIONAL LAYER 1:
	# 	# Remember, depth, height, width, in channels, out channels.
	# 	# self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size,self.conv1_size,1,self.conv1_num_filters],stddev=0.1),name='W_conv1')
	# 	# Introducing RGB channel:
	# 	self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size,self.conv1_size,3,self.conv1_num_filters],stddev=0.1),name='W_conv1')
	# 	self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[self.conv1_num_filters]),name='b_conv1')

	# 	self.conv1 = tf.nn.conv3d(self.input,self.W_conv1,strides=[1,self.conv1_stride,self.conv1_stride,self.conv1_stride,1],padding='SAME') + self.b_conv1
	# 	self.relu_conv1 = tf.nn.relu(self.conv1)

	# 	# DEFINING CONVOLUTIONAL LAYER 2: 		
	# 	# Conv layer 2: 
	# 	self.W_conv2 = tf.Variable(tf.truncated_normal([self.conv2_size,self.conv2_size,self.conv2_size,self.conv1_num_filters,self.conv2_num_filters],stddev=0.1),name='W_conv2')
	# 	self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[self.conv2_num_filters]),name='b_conv2')

	# 	self.conv2 = tf.nn.conv3d(self.relu_conv1,self.W_conv2,strides=[1,self.conv2_stride,self.conv2_stride,self.conv2_stride,1],padding='SAME') + self.b_conv2
	# 	self.relu_conv2 = tf.nn.relu(self.conv2)

	# 	# DEFINING CONVOLUTIONAL LAYER 3: 
	# 	# Output layer 3:
	# 	self.W_conv3 = tf.Variable(tf.truncated_normal([self.conv3_size,self.conv3_size,self.conv3_size,self.conv2_num_filters,self.conv3_num_filters],stddev=0.1),name='W_conv3')
	# 	self.b_conv3 = tf.Variable(tf.constant(0.1,shape=[self.conv3_num_filters]),name='b_conv3')

	# 	# Reward is the "output of this convolutional layer.	"
	# 	# self.reward = tf.nn.conv3d(self.relu_conv2,self.W_conv3,strides=[1,self.conv3_stride,self.conv3_stride,self.conv3_stride,1],padding='SAME') + self.b_conv3
	# 	# Converting to downsampling.
	# 	self.reward = tf.add(tf.nn.conv3d(self.relu_conv2,self.W_conv3,strides=[1,self.conv3_stride,self.conv3_stride,self.conv3_stride,1],padding='VALID'),self.b_conv3,name='reward')
	# 	# NOW THE REWARD SHOULD BE OF SHAPE: 
	# 	# <discrete_z, discrete_y, discrete_x, action_size*discrete_theta>

	# 	self.reward_reshape = tf.reshape(self.reward,shape=[-1,self.discrete_z, self.discrete_y, self.discrete_x, self.discrete_theta, self.action_size],name='reshaped_reward')

	# 	# IGNORING RLNN's VIRCNN unit for now; setting pre_Qvalues to a placeholder that will be fed zeroes.
	# 		# OPTION 1: ACTION_SIZE*DISCRETE_THETA for final channel
	# 	# self.pre_Qvalues = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y, self.discrete_x, self.action_size*self.discrete_theta],name='pre_Qvalues')
	# 		# OPTION 2: , DISCRETE_THETA, ACTION_SIZE. Remember, as long as you have consistent reshaping, the order doesn't matter.
	# 	self.pre_Qvalues = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y, self.discrete_x, self.discrete_theta, self.action_size],name='pre_Qvalues')

	# 	# Computing Q values (across the entire space) as sum of reward and pre_Qvalues.
	# 	# self.Qvalues = tf.add(self.reward,self.pre_Qvalues,name='compute_Qvalues')
	# 	self.Qvalues = tf.add(self.reward_reshape,self.pre_Qvalues,name='compute_Qvalues')

	# 	# Creating placeholder for belief. 
	# 		# 5 dimensional for potentially batches, and so multiply can broadcast size with Q values.
	# 	# self.belief = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y,self.discrete_x,1],name='belief')
	# 		# 6 dimensional for potentially batches., now this is z,y,x,theta.
	# 	self.belief = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y,self.discrete_x,self.discrete_theta,1],name='belief')
	
	# 	# Computing belief space Q Values.
	# 	# tf.multiply supports broadcasting. The reduce sum should be along x,y and z, but not batches and action channel.
	# 	self.belief_space_Qvalues = tf.reduce_sum(tf.multiply(self.belief,self.Qvalues),axis=[1,2,3,4],name='compute_belief_space_Qvalues')

	# 	# DON'T NEED TO EXPLICITLY COMPUTE SOFTMAX. tf.nn.softmax_cross_entropy...
	# 	# # Softmax over belief space Q values.
	# 	# self.softmax_belQ = tf.nn.softmax(self.belief_space_Qvalues)

	# 	# Placeholder for targets.
	# 	self.target_beta = tf.placeholder(tf.float32,shape=[None,self.action_size],name='target_actions')

	# 	# Computing the loss: 
	# 	# THIS IS THE CROSS ENTROPY LOSS. 	
	# 	self.loss = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels=self.target_beta,logits=self.belief_space_Qvalues))

	# 	# CREATING TRAINING VARIABLES:
	# 	self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='Adam_Optimizer')

	# 	# CREATING SUMMARIES:
	# 	self.loss_summary = tf.summary.scalar('Loss',self.loss)
	# 	self.merged = tf.summary.merge_all()

	# 	self.saver = tf.train.Saver(max_to_keep=None)

	# 	init = tf.global_variables_initializer()
	# 	self.sess.run(init)

	# def load_trajectory(self, traj, actions):

	# 	# Assume the trajectory file has positions and velocities.
	# 	# self.orig_traj = traj[0:len(traj):5,:]
	# 	# self.orig_vel = actions[0:len(traj):5,:]
	# 	self.orig_traj = traj
	# 	self.orig_vel = actions

	# 	self.interp_traj = npy.zeros((len(self.orig_traj),8,3),dtype='int')
	# 	self.interp_traj_percent = npy.zeros((len(self.orig_traj),8))

	# 	self.interp_vel = npy.zeros((len(self.orig_traj),3,3),dtype='int')
	# 	self.interp_vel_percent = npy.zeros((len(self.orig_traj),3))
		
	# 	self.preprocess_canonical()
	# 	self.initialize_pointset()

	def load_trajectory(self, traj, orientation):

		# Assume the trajectory file has positions and velocities
		self.orig_traj = traj
		self.orig_vel = npy.diff(self.orig_traj,axis=0)
		self.orig_traj = self.orig_traj[:len(self.orig_vel),:]

		self.orig_orient = orientation
		unwrapped = npy.unwrap(orientation)
		self.orig_angular_vel = npy.diff(unwrapped,axis=0)
		self.orig_orient = self.orig_orient[:len(self.orig_angular_vel)]

		# Linear trajectory interpolation and velocity interpolation array.
		self.interp_traj = npy.zeros((len(self.orig_traj),8,3),dtype='int')
		self.interp_traj_percent = npy.zeros((len(self.orig_traj),8))

		self.interp_vel = npy.zeros((len(self.orig_vel),3,3),dtype='int')
		self.interp_vel_percent = npy.zeros((len(self.orig_traj),3))
		
		# Angular trajectory interplation and velocity interpolation arrays.
		self.interp_angular_traj = npy.zeros((len(self.orig_traj),2),dtype='int')
		self.interp_angular_percent = npy.zeros((len(self.orig_traj),2))

		self.interp_angular_vel = npy.zeros((len(self.orig_vel)),dtype='int')
		self.interp_angular_vel_percent = npy.zeros((len(self.orig_vel),2))

		self.preprocess_canonical()
		self.preprocess_angular()
		self.initialize_pointset()

	def initialize_pointset(self):
		# Initializing set of  sampling points for Gaussian Observation Kernel.
		self.pointset = npy.zeros((64,3))
		add = [-1,0,1,2]

		for i in range(4):
			for j in range(4):
				for k in range(4):
					self.pointset[16*i+4*j+k] = [add[i],add[j],add[k]]

		self.alter_point_set = (self.pointset*self.grid_cell_size).reshape(4,4,4,3)
		self.angular_pointset = npy.array(add)

	def load_transition(self,trans):
		# Loading the transition model learnt from BPRCNN.
		self.trans = trans

	def _powerset(self, iterable):
		# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

	def angular_interpolate_coefficients(self, angular_state):

		base_indices = npy.floor((angular_state-self.ang_traj_lower)/self.angular_grid_cell_size)
		base_point = self.angular_grid_cell_size*npy.floor(angular_state/self.angular_grid_cell_size)
		base_lengths = angular_state - base_point
		bases = []

		bases.append((base_lengths/self.angular_grid_cell_size,base_indices))		

		index_to_add = (base_indices+1)%self.discrete_theta
		bases.append((1.-base_lengths/self.angular_grid_cell_size,index_to_add))

		return bases

	def interpolate_coefficients(self, point, traj_or_action=1):
		# VARIABLE GRID SIZE ALONG DIFFERENT DIMENSIONS:

		# Choose whether we are interpolating a trajectory or an action data point.

		# Now lower is just uniformly -1. 
		lower = -npy.ones(3)

		# If traj_or_action is 0, it's an action, if 1, it's a trajectory.
		# If trajectory, z lower must be 0.
		lower[2] += traj_or_action

		# grid_cell_size = traj_or_action * self.traj_cell + (1-traj_or_action)*self.action_cell
		grid_cell_size = traj_or_action*self.grid_cell_size + (1-traj_or_action)*self.action_cell

		base_indices = npy.floor((point-lower)/grid_cell_size)
		base_point = grid_cell_size*npy.floor(point/grid_cell_size)		
		base_lengths = point - base_point
		bases = []

		for index_set in self._powerset(range(self.dimensions)):
			index_set = set(index_set)
			volume = 1 
			# point_to_add = base_point.copy()
			index_to_add = base_indices.copy()

			for i in range(self.dimensions):
				if i in index_set:
					side_length = base_lengths[i]			
					# point_to_add += self.grid_cell_size[i]
					index_to_add[i] += 1
				else:
					side_length = grid_cell_size[i] - base_lengths[i]

				volume *= side_length / grid_cell_size[i]

			# bases.append((volume, point_to_add, index_to_add))			
			bases.append((volume, index_to_add))			

		return bases

	# def construct_from_ext_state(self):

	# 	w = self.w
	# 	self.from_state_ext[w:self.discrete_x+w,w:self.discrete_y+w,w:self.discrete_z+w] = copy.deepcopy(self.from_state_belief)

	def construct_from_ext_state(self):

		w = self.w
		self.from_state_ext[w:self.discrete_x+w,w:self.discrete_y+w,w:self.discrete_z+w] = copy.deepcopy(self.from_state_belief)

		# NOW USING EXTENDED ANGULAR STATE TO CONVERT THE CIRCULAR CONVOLUTION TO A LINEAR CONVOLUTION
		self.from_angular_ext[w:self.discrete_theta+w] = copy.deepcopy(self.from_angular_belief)
		for j in range(w):
			self.from_angular_ext[w-j-1] = self.from_angular_belief[-1-j]
			self.from_angular_ext[self.discrete_theta+w+j] = self.from_angular_belief[j]

	def belief_prediction(self):
		# Implements the motion update of the Bayes Filter.

		w = self.w
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		# Linear then angular
		self.to_state_ext[:,:,:] = 0

		self.intermed_angular_belief[:] = 0

		for k in range(self.action_size):
			self.to_state_ext += self.beta[k]*signal.convolve(self.from_state_ext,self.trans[k],'same')

		for k in range(self.angular_action_size):		
			# self.intermed_angular_belief += self.angular_beta[k]*convolve1d(self.from_angular_belief,self.angular_trans[k],mode='wrap')			
			self.intermed_angular_belief += self.angular_beta[k]*signal.convolve(self.from_angular_ext,self.angular_trans[k],'valid')			

		# Folding over the extended belief:
		for i in range(w):			
			# Linear folding
			self.to_state_ext[i+1,:,:] += self.to_state_ext[i,:,:]
			self.to_state_ext[i,:,:] = 0
			self.to_state_ext[:,i+1,:] += self.to_state_ext[:,i,:]
			self.to_state_ext[:,i,:] = 0
			self.to_state_ext[:,:,i+1] += self.to_state_ext[:,:,i]
			self.to_state_ext[:,:,i] = 0

			self.to_state_ext[dx+2*w-i-2,:,:] += self.to_state_ext[dx+2*w-i-1,:,:]
			self.to_state_ext[dx+2*w-i-1,:,:] = 0
			self.to_state_ext[:,dy+2*w-i-2,:] += self.to_state_ext[:,dy+2*w-i-1,:]
			self.to_state_ext[:,dy+2*w-i-1,:] = 0
			self.to_state_ext[:,:,dz+2*w-i-2] += self.to_state_ext[:,:,dz+2*w-i-1]
			self.to_state_ext[:,:,dz+2*w-i-1] = 0

		# Don't skip this for translational beliefs.
		self.intermed_belief = copy.deepcopy(self.to_state_ext[w:dx+w,w:dy+w,w:dz+w])
		self.intermed_belief /= self.intermed_belief.sum()

	def belief_correction(self):
		# Implements the Bayesian Observation Fusion to Correct the Predicted / Intermediate Belief.

		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z
		
		h = self.h
		obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size).astype(int)
		angular_obs = npy.floor((self.angular_observed_state - self.ang_traj_lower)/self.angular_grid_cell_size).astype(int)

		# UPDATING TO THE NEW GAUSSIAN KERNEL OBSERVATION MODEL:
		self.extended_obs_belief[:,:,:] = 0.
		self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h] = self.intermed_belief		
		self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3] = npy.multiply(self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)

		self.extended_angular_obs_belief[:] = 0.
		self.extended_angular_obs_belief[h:self.discrete_theta+h] = self.intermed_angular_belief

		self.extended_angular_obs_belief[h+angular_obs[0]-1:h+angular_obs[0]+3] = npy.multiply(self.extended_angular_obs_belief[h+angular_obs[0]-1:h+angular_obs[0]+3],self.angular_obs_model)

		# # Actually obs[0]-h:obs[0]+h, but with extended belief, we add another h:
		# self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)		

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h])
		self.to_state_belief /= self.to_state_belief.sum() 

		self.to_angular_belief = copy.deepcopy(self.extended_angular_obs_belief[h:self.discrete_theta+h])
		self.to_angular_belief /= self.to_angular_belief.sum()

	def map_triplet_to_action_canonical(self,triplet):

		if triplet[0]==-1:
			return 0
		if triplet[0]==1:
			return 1
		if triplet[1]==-1:
			return 2
		if triplet[1]==1:
			return 3
		if triplet[2]==-1:
			return 4
		if triplet[2]==1:
			return 5

	def map_singlet_to_angular_action(self, singlet):

		if singlet==-1:
			return 0
		if singlet==1:
			return 1

	def preprocess_angular(self):
		print("Preprocessing Angular Data.")

		# Loads angles from -pi to pi.
		norm_vector = npy.pi

		self.orig_orient /= norm_vector

		vel_norm_vector = npy.max(abs(self.orig_angular_vel),axis=0)
		self.orig_angular_vel /= vel_norm_vector

		for t in range(len(self.orig_orient)):

			split = self.angular_interpolate_coefficients(self.orig_orient[t])
			count = 0

			for percent, indices in split:
				
				self.interp_angular_traj[t,count] = indices
				self.interp_angular_percent[t,count] = percent
				count +=1

			ang_vel = self.orig_angular_vel[t]

			self.interp_angular_vel[t] = abs(ang_vel)/ang_vel

			r = self.interp_angular_vel[t].copy()
			r += 1
			r /= 2
			
			self.interp_angular_vel_percent[t,r] = abs(ang_vel)

		npy.save("Interp_Yaw.npy",self.interp_angular_traj)
		npy.save("Interp_Yaw_Percent.npy",self.interp_angular_percent)
		npy.save("Interp_YawRate.npy",self.interp_angular_vel)
		npy.save("Interp_YawRate_Percent.npy",self.interp_angular_vel_percent)

	def preprocess_canonical(self):
		print("Preprocessing the Data.")

		# Normalize trajectory.
		# norm_vector = [2.5,2.5,1.]		
		norm_vector = [1.,1.,1.]		
		self.orig_traj /= norm_vector

		# Normalize actions (velocities).
		self.orig_vel /= norm_vector
		vel_norm_vector = npy.max(abs(self.orig_vel),axis=0)
		self.orig_vel /= vel_norm_vector

		for t in range(len(self.orig_traj)):
			
			# Trajectory. 
			split = self.interpolate_coefficients(self.orig_traj[t],1)
			count = 0
			for percent, indices in split: 
				self.interp_traj[t,count] = indices
				self.interp_traj_percent[t,count] = percent
				count += 1

			# Action. 
			vel = self.orig_vel[t]/npy.linalg.norm(self.orig_vel[t])

			self.interp_vel[t,0] = [abs(vel[0])/vel[0],0,0]
			self.interp_vel[t,1] = [0,abs(vel[1])/vel[1],0]
			self.interp_vel[t,2] = [0,0,abs(vel[2])/vel[2]]
			self.interp_vel_percent[t] = abs(vel)

			# Forcing percentages to sum to 1:
			self.interp_vel_percent[t] /= self.interp_vel_percent[t].sum()

		npy.save("Interp_Traj.npy",self.interp_traj)
		npy.save("Interp_Vel.npy",self.interp_vel)
		npy.save("Interp_Traj_Percent.npy",self.interp_traj_percent)
		npy.save("Interp_Vel_Percent.npy",self.interp_vel_percent)

	def parse_data(self,timepoint):
		# Setting from state belief from interp_traj.
		# For each of the 8 grid points, set the value of belief = percent at that point. 
		# This should sum to 1.
		self.beta[:] = 0.
		self.angular_beta[:] = 0.

		self.target_belief[:,:,:] = 0.
		self.from_state_belief[:,:,:] = 0.

		self.target_angular_belief[:] = 0.
		self.from_angular_belief[:] = 0.

		for k in range(8):
			# Here setting the from state; then call construct extended. 
			self.from_state_belief[self.interp_traj[timepoint,k,0],self.interp_traj[timepoint,k,1],self.interp_traj[timepoint,k,2]] = self.interp_traj_percent[timepoint,k]

		for k in range(2):
			self.from_angular_belief[self.interp_angular_traj[timepoint,k]] = self.interp_angular_percent[timepoint,k]

		# Setting beta: This becomes the targets in Cross Entropy.
		# Map triplet indices to action index, set that value of beta to percent.
		for k in range(3):
			self.beta[self.map_triplet_to_action_canonical([self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2]])] = self.interp_vel_percent[timepoint,k] 

		for k in range(2):
			self.angular_beta[k] = self.interp_angular_vel_percent[timepoint,k]

		# Updating action counter of how many times each action was taken; not as important in the QMDP RCNN as BPRCNN.
		# self.action_counter += self.beta

		# MUST ALSO PARSE AND LOAD INPUT POINTCLOUDS.
		self.observed_state = self.orig_traj[timepoint]
		self.angular_observed_state = self.orig_orient[timepoint]

		mean = self.observed_state - self.grid_cell_size*npy.floor(self.observed_state/self.grid_cell_size)
		self.obs_model = mvn.pdf(self.alter_point_set,mean=mean,cov=0.005)
		self.obs_model /= self.obs_model.sum()

		angular_mean = self.angular_observed_state - self.angular_grid_cell_size*npy.floor(self.angular_observed_state/self.angular_grid_cell_size)
		self.angular_obs_model = mvn.pdf(self.angular_pointset,mean=angular_mean,cov=0.005)
		self.angular_obs_model /= self.angular_obs_model.sum()

	def recurrence(self):
		# Recurrence in the BP RCNN.
		self.from_state_belief = self.to_state_belief.copy()

	def update_QMDP_values(self):

		for k in range(self.action_size):
			self.belief_space_q[k] = npy.sum(self.Qvalues[k]*self.from_state_belief)

	def calc_softmax_beta(self):
		# Calculate the softmax values of the belief space Q values.
		self.softmax_q = npy.exp(self.belief_space_q)/npy.sum(npy.exp(self.belief_space_q))

	def update_Q_estimate(self, decay_factor=1.0):
		self.Qvalues = (1-decay_factor)*self.reward + decay_factor*self.Qvalues

	def backprop_reward(self, timer):
		self.update_QMDP_values()
		self.calc_softmax_beta()

		alpha = self.learning_rate - timer*self.annealing_rate

		self.backprop_belief = npy.outer(self.from_state_belief,self.from_angular_belief).reshape((self.discrete_x,self.discrete_y,self.discrete_z,self.discrete_theta))
		# self.backprop_belief = npy.outer(self.from_state_belief,self.from_angular_belief).reshape((self.discrete_x,self.discrete_y,self.discrete_z,self.discrete_theta))

		for k in range(self.action_size):
			self.reward[k] -= alpha * (self.softmax_q[k] - self.beta[k])*self.backprop_belief

	def conv_layer(self):

		# Now modifying convolutional layer of the VI RCNN to take into account that yaw wraps around. 

		# Construct extended value function - must extended along every dimension, then implement as a 4D Valid conv, instead of Same conv.
		w = 1
		self.extended_value = npy.zeros((self.discrete_x+2*w, self.discrete_y+2*w, self.discrete_z+2*w, self.discrete_theta+2*w))
		self.extended_value[w:-w,w:-w,w:-w,w:-w] = self.value_function
		
		self.extended_value[w:-w,w:-w,w:-w,0] = self.value_function[:,:,:,-1]
		self.extended_value[w:-w,w:-w,w:-w,-1] = self.value_function[:,:,:,0]	

		# Convolve.
		for k in range(self.action_size):
			self.Qvalues[k] = signal.convolve(self.extended_value,self.trans[k],'valid')

	def max_pool(self):
		self.value_function = npy.amax(self.Qvalues,axis=0)
		self.policy = npy.argmax(self.Qvalues,axis=0)

	def feedback(self):
		self.max_pool()
		self.conv_layer()

	def train_timepoint(self,timepoint):

		# Parse Data:
		self.parse_data(timepoint)

		# PROCESSING BELIEFS OUTSIDE TENSORFLOW
		# Construct the from_extended_state for belief propagation.
		self.construct_from_ext_state()
		
		# Propagate belief: Convolve with Trans model and merge intermediate beliefs.
		self.belief_prediction()
		# Correct Intermediate Belief (Observation Fusion)
		self.belief_correction()

		# CODE FOR BACKPROP OUTSIDE TENSORFLOW; DON'T USE FOR DEEP REWARDS
		# # Backpropagate the Cross Entropy / Negative Log Likelihood. 
		# # Equivalent to the KL Divergence; since the target distribution is fixed.
		self.backprop_reward(num_epochs)

		# # Update Q Values: This is different from Feedback
		self.update_Q_estimate(0.99)

		# # Recurrence. 
		self.recurrence()

		# TENSORFLOW TRAINING:
		# Remember, must feed: input <--corresponding point cloud, belief <-- to_state_belief, target_beta <-- beta, pre_Qvalues <-- 0. 
		# DO ALL RESHAPING, TRANSPOSING HERE.


	def train_QMDPRCNN(self,file_index):

		
		for e in range(self.epochs):
			print("Training Epoch:",e)

			for j in range(len(self.interp_traj)-1):
				print("Training: File: {0}, Epoch: {1}, Time Step: {2}.".format(file_index,e,j))

				# CURRENTLY TRAINING STOCHASTICALLY: NO BATCHES.
				reward_val = self.train_timepoint(j)

			self.feedback()
			self.save_model(reward_val)

			self.saver.save(self.sess,"Model_{0}.ckpt".format(e))

	def save_model(self,reward_val):
		# Now, we have to save the TensorFlow model instead.		

		# npy.save("Learnt_Reward_TF.npy",self.reward.eval(session=self.sess))
		# print("Saving the Model.")
		# saver.save(self.sess,"Model")

		npy.save("Learnt_Reward_TF.npy",reward_val)

def main(args):

	# Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="1,2")
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)


	# sess = tf.Session()
	# Create an instance of QMDP_RCNN class. 
	qmdprcnn = QMDP_RCNN()

	# Initialize the TensorFlow model.
	qmdprcnn.initialize_tensorflow_model(sess)

	# Load the data to train on:
	traj = npy.load(str(sys.argv[1]))
	# actions = npy.load(str(sys.argv[2]))
	orient = npy.load(str(sys.argv[2]))
	trans = npy.load(str(sys.argv[3]))

	qmdprcnn.load_transition(trans)

	# Train:
	# for i in range(1):
	i = 1
	# qmdprcnn.load_trajectory(traj[i][68:],actions[i][67:])
	qmdprcnn.load_trajectory(traj[i][68:],orient[i][68:])
	qmdprcnn.train_QMDPRCNN(i)

if __name__ == '__main__':
	main(sys.argv)