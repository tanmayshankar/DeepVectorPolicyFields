#!/usr/bin/env python
from headers import *
from variables import *

class BPRCNN():

	def __init__(self):

		# Defining common variables.
		self.discrete = 50
		self.dimensions = 3
		self.action_size = 6
		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		# Defining transition model.
		self.trans_space = 3
		self.trans = npy.random.random((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		
		# Defining observation model.
		self.obs_space = 5
		
		# Defining Value function, policy, etc. 
		self.value_function = npy.zeros((self.discrete, self.discrete, self.discrete))
		self.policy = npy.zeros((self.discrete, self.discrete, self.discrete))

		# Discount
		self.gamma = 0.95

		# Defining belief variables.
		self.from_state_belief = npy.zeros((self.discrete,self.discrete,self.discrete))
		self.to_state_belief = npy.zeros((self.discrete,self.discrete,self.discrete))
		self.target_belief = npy.zeros((self.discrete,self.discrete,self.discrete))
		# self.corrected_to_state_belief = npy.zeros((self.discrete,self.discrete,self.discrete))
		self.intermed_belief = npy.zeros((self.discrete,self.discrete,self.discrete))

		# Defining extended belief states. 
		self.w = self.trans_space/2
		self.to_state_ext = npy.zeros((self.discrete+2*self.w,self.discrete+2*self.w,self.discrete+2*self.w))
		self.from_state_ext = npy.zeros((self.discrete+2*self.w,self.discrete+2*self.w,self.discrete+2*self.w))

		# Defining trajectory
		self.traj = []
		self.actions = []
		self.beta = npy.zeros(self.action_size)

		# Defining observation model related variables. 
		self.obs_space = 3
		self.obs_model = npy.zeros((self.obs_space,self.obs_space,self.obs_space))
		self.h = self.obs_space/2
		self.extended_obs_belief = npy.zeros((self.obs_space+self.h*2,self.obs_space+self.h*2,self.obs_space+self.h*2))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 0


	def load_trajectory(self, traj, actions):

		# Assume the trajectory file has positions and velocities.
		self.traj = traj
		self.actions = actions

	def construct_from_ext_state(self):

		w = self.w
		d = self.discrete
		self.from_state_ext[w:d+w,w:d+w,w:d+w] = copy.deepcopy(self.from_state_ext)

	# def motion_update(self):

	def update_beta(self, act):
		# Update the beta values in order to predict intermediate belief. 

		# beta : LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP.
		# Assuming 6 cardinal directions for now. May need to change to Trilinear interpolation. 
		act /= npy.linalg.norm(act)		
		self.beta = npy.array([act[0]*(act[0]<0),act[0]*(act[0]>=0),act[1]*(act[1]<0),act[1]*(act[1]>=0),act[2]*(act[2]<0),act[2]*(act[2]>=0)])

	def belief_prediction(self):
		# Implements the motion update of the Bayes Filter.

		w = self.w
		d = self.discrete

		self.to_state_ext[:,:,:] = 0
		self.update_beta()

		for k in range(self.action_size):
			self.to_state_ext += beta[k]*signal.convolve(self.from_state_ext,self.trans[k],'same')

		# Folding over the extended belief:
		for i in range(w):			
			self.to_state_ext[i+1,:,:] += self.to_state_ext[i,:,:]
			self.to_state_ext[i,:,:] = 0
			self.to_state_ext[:,i+1,:] += self.to_state_ext[:,i,:]
			self.to_state_ext[:,i,:] = 0
			self.to_state_ext[:,:,i+1] += self.to_state_ext[:,:,i]
			self.to_state_ext[:,:,i] = 0

			self.to_state_ext[d+2*w-i-2,:,:] += self.to_state_ext[d+2*w-i-1,:,:]
			self.to_state_ext[d+2*w-i-1,:,:] = 0
			self.to_state_ext[:,d+2*w-i-2,:] += self.to_state_ext[:,d+2*w-i-1,:]
			self.to_state_ext[:,d+2*w-i-1,:] = 0
			self.to_state_ext[:,:,d+2*w-i-2] += self.to_state_ext[d+2*w-i-1,:,:]
			self.to_state_ext[:,:,d+2*w-i-1] = 0
		
		self.intermed_belief = copy.deepcopy(self.to_state_ext[w:d+w,w:d+w,w:d+w])

	def belief_correction(self):
		# Implements the Bayesian Observation Fusion to Correct the Predicted / Intermediate Belief.

		d = self.discrete
		h = self.h
		obs = self.observed_state
		
		self.extended_obs_belief[h:d+h,h:d+h,h:d+h] = self.intermed_belief
		self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:d+h,h:d+h,h:d+h])
		self.to_state_belief /= self.to_state_belief.sum()

	def compute_sensititives(self):

		# Compute the sensitivity values. 
		self.sensitivity = self.target_belief - self.from_state_belief

		obs = self.observed_state
		h = self.h

		# Redefining the sensitivity values to be (target_b - pred_b)*obs_model = F.
		self.sensitivity = npy.multiply(self.sensitivity[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)

	def backprop_convolution(self):

		self.compute_sensititives()
		w = self.w
		
		# Set learning rate and lambda value.
		
		self.time_count +=1
		alpha = self.learning_rate - self.annealing_rate*self.time_count

		# Calculate basic gradient update. 
		grad_update = -2*signal.convolve(self.from_state_ext,self.sensitivity, 'valid')
		
		t0 = npy.zeros((self.trans_space,self.trans_space,self.trans_space))
		t1 = npy.ones((self.trans_space,self.trans_space,self.trans_space))

		for k in range(self.action_size):
			act_grad_update = self.beta[k]*(grad_update + self.lamda*self.trans[k].sum() -1.)
			self.trans[k] = npy.maximum(t0,npy.minimum(t1,self.trans[k]-alpha*act_grad_update))

			# Is this needed? 
			# self.trans[k] /= self.trans[k].sum()

	def recurrence(self):
		# With Teacher Forcing: Setting next belief to the previous target / ground truth.
		self.from_state_belief = copy.deepcopy(self.target_belief)

	def parse_data(self):
		

	def train_BPRCNN(self):

		# Parse Data:
			# Set target belief, beta, etc...
		self.parse_data()

		# Construct the from_extended_state for belief propagation
		self.construct_from_ext_state()
		
		# Propagate belief: 
			# Convolve with Trans model and merge intermediate beliefs.
		self.belief_prediction()
			# Correct Intermediate Belief (Observation Fusion)
		self.belief_correction()

		# Backpropagate with ground truth belief.
		self.backprop_convolution()

		# Recurrence. 
		self.recurrence()




