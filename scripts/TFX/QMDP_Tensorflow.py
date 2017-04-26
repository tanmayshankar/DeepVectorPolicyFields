#!/usr/bin/env python 
from headers import *
import tensorflow as tf

class QMDP_RCNN():

	def __init__(self):

		self.discrete_x = 50
		self.discrete_y = 50
		self.discrete_z = 32
		self.action_size = 6

		self.input_x = 51
		self.input_y = 51
		self.input_z = 11

		self.input_x = 101
		self.input_y = 101
		self.input_z = 66

		self.conv1_size = 3
		self.conv2_size = 3
		self.conv3_size = 3

		self.conv1_stride = 1
		self.conv2_stride = 1
		self.conv3_stride = 2

		self.conv1_num_filters = 20
		self.conv2_num_filters = 30
		self.conv3_num_filters = 6

		self.dimensions = 3

		# Setting discretization variables
		self.action_lower = -1
		self.action_upper = +1
		self.action_cell = npy.ones(self.dimensions)

		# Assuming lower is along all dimensions. 		
		self.traj_lower = npy.array([-1,-1,0])
		self.traj_upper = +1
		
		self.grid_cell_size = npy.array((self.traj_upper-self.traj_lower)).astype(float)/[self.discrete_x-1, self.discrete_y-1, self.discrete_z-1]		

		# SHIFTING BACK TO CANONICAL ACTIONS
		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		self.trans_space = 3
		# Remember to LOAD transition model. 

		self.action_counter = npy.zeros(self.action_size)

		# Defining observation model.
		self.obs_space = 5
		
		# DON'T NEED THESE; DEFINED IN TENSORFLOW MODEL.
		# # Defining Value function, policy, etc. 	
		# self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z))
		# self.policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z))
		# self.reward = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))
		# self.Qvalues = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))

		# # Defining belief space Q values; softmax values. 
		# self.belief_space_q = npy.zeros(self.action_size)
		# self.softmax_q = npy.zeros(self.action_size)

		# Discount
		self.gamma = 0.95

		# Defining dummy input volume.
		# Introducing RGB
		self.input_volume = npy.ones((3,self.input_x,self.input_y,self.input_z))

		# Defining belief variables.
		self.from_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.to_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.target_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.intermed_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))

		# Defining extended belief states. 
		self.w = self.trans_space/2
		self.to_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))
		self.from_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))

		# Defining trajectory
		self.orig_traj = []
		self.orig_vel = []

		self.beta = npy.zeros(self.action_size)

		# Defining observation model related variables. 
		self.obs_space = 4
		self.obs_model = npy.zeros((self.obs_space,self.obs_space,self.obs_space))
		# Apparently it was only /2 for obs_space=3. Actually should be -1. 
		self.h = self.obs_space-1

		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))

		self.dummy_zeroes = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z, self.action_size))
		# # Setting hyperparameters
		# self.time_count = 0
		# self.lamda = 1
		# self.learning_rate = 0.5
		# self.annealing_rate = 0.1

		# Setting training parameters: 
		self.epochs = 20

	def initialize_tensorflow_model(self,sess):

		# Initializing Tensorflow Session: 
		self.sess = sess

		# Remember, TensorFlow wants things as: Batch / Depth / Height / Width / Channels.
		# self.input = tf.placeholder(tf.float32,shape=[None,self.input_z,self.input_y,self.input_x,1],name='input')
		# Introducing RGB channel: 
		self.input = tf.placeholder(tf.float32,shape=[None,self.input_z,self.input_y,self.input_x,3],name='input')

		# self.reward = tf.placeholder(tf.float32,shape=[None,self.discrete_x,self.discrete_y,self.discrete_z],name='reward')

		# DEFINING CONVOLUTIONAL LAYER 1:
		# Remember, depth, height, width, in channels, out channels.
		# self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size,self.conv1_size,1,self.conv1_num_filters],stddev=0.1),name='W_conv1')
		# Introducing RGB channel:
		self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size,self.conv1_size,3,self.conv1_num_filters],stddev=0.1),name='W_conv1')
		self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[self.conv1_num_filters]),name='b_conv1')

		self.conv1 = tf.nn.conv3d(self.input,self.W_conv1,strides=[1,self.conv1_stride,self.conv1_stride,self.conv1_stride,1],padding='SAME') + self.b_conv1
		self.relu_conv1 = tf.nn.relu(self.conv1)

		# DEFINING CONVOLUTIONAL LAYER 2: 		
		# Conv layer 2: 
		self.W_conv2 = tf.Variable(tf.truncated_normal([self.conv2_size,self.conv2_size,self.conv2_size,self.conv1_num_filters,self.conv2_num_filters],stddev=0.1),name='W_conv2')
		self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[self.conv2_num_filters]),name='b_conv2')

		self.conv2 = tf.nn.conv3d(self.relu_conv1,self.W_conv2,strides=[1,self.conv2_stride,self.conv2_stride,self.conv2_stride,1],padding='SAME') + self.b_conv2
		self.relu_conv2 = tf.nn.relu(self.conv2)

		# DEFINING CONVOLUTIONAL LAYER 3: 
		# Output layer 3:
		self.W_conv3 = tf.Variable(tf.truncated_normal([self.conv3_size,self.conv3_size,self.conv3_size,self.conv2_num_filters,self.conv3_num_filters],stddev=0.1),name='W_conv3')
		self.b_conv3 = tf.Variable(tf.constant(0.1,shape=[self.conv3_num_filters]),name='b_conv3')

		# Reward is the "output of this convolutional layer.	"
		# self.reward = tf.nn.conv3d(self.relu_conv2,self.W_conv3,strides=[1,self.conv3_stride,self.conv3_stride,self.conv3_stride,1],padding='SAME') + self.b_conv3
		# Converting to downsampling.
		self.reward = tf.nn.conv3d(self.relu_conv2,self.W_conv3,strides=[1,self.conv3_stride,self.conv3_stride,self.conv3_stride,1],padding='VALID') + self.b_conv3

		# IGNORING RLNN's VIRCNN unit for now; setting pre_Qvalues to a placeholder that will be fed zeroes.
		self.pre_Qvalues = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y, self.discrete_x, self.action_size],name='pre_Qvalues')

		# Computing Q values (across the entire space) as sum of reward and pre_Qvalues.
		self.Qvalues = tf.add(self.reward,self.pre_Qvalues,name='compute_Qvalues')

		# Creating placeholder for belief. 
		# 5 dimensional for potentially batches, and so multiply can broadcast size with Q values.
		self.belief = tf.placeholder(tf.float32,shape=[None,self.discrete_z,self.discrete_y,self.discrete_x,1],name='belief')
	
		# Computing belief space Q Values.
		# tf.multiply supports broadcasting. The reduce sum should be along x,y and z, but not batches and action channel.
		self.belief_space_Qvalues = tf.reduce_sum(tf.multiply(self.belief,self.Qvalues),axis=[1,2,3],name='compute_belief_space_Qvalues')

		# DON'T NEED TO EXPLICITLY COMPUTE SOFTMAX. tf.nn.softmax_cross_entropy...
		# # Softmax over belief space Q values.
		# self.softmax_belQ = tf.nn.softmax(self.belief_space_Qvalues)

		# Placeholder for targets.
		self.target_beta = tf.placeholder(tf.float32,shape=[None,self.action_size],name='target_actions')

		# Computing the loss: 
		# THIS IS THE CROSS ENTROPY LOSS. 	
		self.loss = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels=self.target_beta,logits=self.belief_space_Qvalues))

		# CREATING TRAINING VARIABLES:
		self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='Adam_Optimizer')

		# CREATING SUMMARIES:
		self.loss_summary = tf.summary.scalar('Loss',self.loss)
		self.merged = tf.summary.merge_all()

		self.saver = tf.train.Saver(max_to_keep=None)

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def load_trajectory(self, traj, actions):

		# Assume the trajectory file has positions and velocities.
		# self.orig_traj = traj[0:len(traj):5,:]
		# self.orig_vel = actions[0:len(traj):5,:]
		self.orig_traj = traj
		self.orig_vel = actions

		self.interp_traj = npy.zeros((len(self.orig_traj),8,3),dtype='int')
		self.interp_traj_percent = npy.zeros((len(self.orig_traj),8))

		self.interp_vel = npy.zeros((len(self.orig_traj),3,3),dtype='int')
		self.interp_vel_percent = npy.zeros((len(self.orig_traj),3))
		
		self.preprocess_canonical()
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

	def load_transition(self,trans):
		# Loading the transition model learnt from BPRCNN.
		self.trans = trans

	def _powerset(self, iterable):
		# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

	def construct_from_ext_state(self):

		w = self.w
		self.from_state_ext[w:self.discrete_x+w,w:self.discrete_y+w,w:self.discrete_z+w] = copy.deepcopy(self.from_state_belief)

	def belief_prediction(self):
		# Implements the motion update of the Bayes Filter.

		w = self.w
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		self.to_state_ext[:,:,:] = 0

		for k in range(self.action_size):
			self.to_state_ext += self.beta[k]*signal.convolve(self.from_state_ext,self.trans[k],'same')

		# Folding over the extended belief:
		for i in range(w):			
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
		
		self.intermed_belief = copy.deepcopy(self.to_state_ext[w:dx+w,w:dy+w,w:dz+w])
		self.intermed_belief /= self.intermed_belief.sum()

	def belief_correction(self):
		# Implements the Bayesian Observation Fusion to Correct the Predicted / Intermediate Belief.

		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z
		
		# h = self.h
		# obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size)

		h = self.h
		obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size).astype(int)

		# UPDATING TO THE NEW GAUSSIAN KERNEL OBSERVATION MODEL:
		self.extended_obs_belief[:,:,:] = 0.
		self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h] = self.intermed_belief		
		self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3] = npy.multiply(self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)

		# # Actually obs[0]-h:obs[0]+h, but with extended belief, we add another h:
		# self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)		

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h])
		self.to_state_belief /= self.to_state_belief.sum()

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
		self.from_state_belief[:,:,:] = 0.

		for k in range(8):
			# Here setting the from state; then call construct extended. 
			self.from_state_belief[self.interp_traj[timepoint,k,0],self.interp_traj[timepoint,k,1],self.interp_traj[timepoint,k,2]] = self.interp_traj_percent[timepoint,k]

		# Setting beta: This becomes the targets in Cross Entropy.
		# Map triplet indices to action index, set that value of beta to percent.
		for k in range(3):
			self.beta[self.map_triplet_to_action_canonical([self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2]])] = self.interp_vel_percent[timepoint,k] 

		# Updating action counter of how many times each action was taken; not as important in the QMDP RCNN as BPRCNN.
		self.action_counter += self.beta

		# Parsing observation model.
		self.observed_state = self.orig_traj[timepoint]
		mean = self.observed_state - self.grid_cell_size*npy.floor(self.observed_state/self.grid_cell_size)
		self.obs_model = mvn.pdf(self.alter_point_set,mean=mean,cov=0.005)
		self.obs_model /= self.obs_model.sum()

		# MUST ALSO PARSE AND LOAD INPUT POINTCLOUDS.

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
		# self.backprop_reward(num_epochs)

		# # Update Q Values: This is different from Feedback
		# self.update_Q_estimate(0.99)

		# # Recurrence. 
		# self.recurrence()

		# TENSORFLOW TRAINING:
		# Remember, must feed: input <--corresponding point cloud, belief <-- to_state_belief, target_beta <-- beta, pre_Qvalues <-- 0. 
		# DO ALL RESHAPING, TRANSPOSING HERE.

		FILE_DIR = "/home/tanmay/Research/DeepVectorPolicyFields/Data/NEW_D2"

		feed_target_beta = self.beta.reshape((1,self.action_size))
		feed_belief = npy.transpose(self.to_state_belief).reshape((1,self.discrete_z,self.discrete_y,self.discrete_x,1))
		
		# feed_input_volume = npy.transpose(self.input_volume).reshape((1,self.input_z,self.input_y,self.input_x,3))
		feed_input_volume = npy.transpose(npy.load(os.path.join(FILE_DIR,"Voxel_TFX_PC{0}.npy".format(timepoint)))).reshape((1,self.input_z,self.input_y,self.input_x,3))

		feed_dummy_zeroes = npy.transpose(self.dummy_zeroes).reshape((1,self.discrete_z,self.discrete_y,self.discrete_x,self.action_size))

		merged_summary, loss_value, reward_val, _ = self.sess.run([self.merged, self.loss, self.reward, self.train], feed_dict={self.input: feed_input_volume, self.target_beta: feed_target_beta, self.belief: feed_belief, self.pre_Qvalues: feed_dummy_zeroes})
		return reward_val

	def train_QMDPRCNN(self,file_index):

		
		for e in range(self.epochs):
			print("Training Epoch:",e)

			for j in range(len(self.interp_traj)-1):
				print("Training: File: {0}, Epoch: {1}, Time Step: {2}.".format(file_index,e,j))

				# CURRENTLY TRAINING STOCHASTICALLY: NO BATCHES.
				reward_val = self.train_timepoint(j)

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
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	
	sess = tf.Session(config=config)

	# Create an instance of QMDP_RCNN class. 
	qmdprcnn = QMDP_RCNN()

	# Initialize the TensorFlow model.
	qmdprcnn.initialize_tensorflow_model(sess)

	# Load the data to train on:
	traj = npy.load(str(sys.argv[1]))
	actions = npy.load(str(sys.argv[2]))
	trans = npy.load(str(sys.argv[3]))

	qmdprcnn.load_transition(trans)

	# Train:
	# for i in range(1):
	i = 1
	qmdprcnn.load_trajectory(traj[i][68:],actions[i][67:])
	qmdprcnn.train_QMDPRCNN(i)

if __name__ == '__main__':
	main(sys.argv)