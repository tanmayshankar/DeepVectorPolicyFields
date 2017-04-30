#!/usr/bin/env python
from headers import *
# from variables import *

class BPRCNN():

	def __init__(self):
		
		# Defining common variables.
		self.discrete_x = 51
		self.discrete_y = 51
		self.discrete_z = 11

		# angular discretization.
		self.discrete_theta = 36
		self.discrete_phi = 19

		self.dimensions = 3
		self.angular_dimensions = 2
		self.action_size = 6

		# Setting discretization variables
		self.action_lower = -1
		self.action_upper = +1
		self.action_cell = npy.ones(self.dimensions)

		# Assuming lower is along all dimensions. 		
		# self.traj_lower = npy.array([-1,-1,0,-1,-1])
		self.traj_lower = npy.array([-1,-1,0])
		self.traj_upper = +1

		self.ang_traj_lower = -1
		self.ang_traj_upper = +1

		# ALWAYS EXPRESSING PHI BEFORE Theta in Indexing
		self.grid_cell_size = npy.array((self.traj_upper-self.traj_lower)).astype(float)/[self.discrete_x-1, self.discrete_y-1, self.discrete_z-1]		
		self.angular_grid_cell_size = npy.array((self.ang_traj_upper-self.ang_traj_lower)).astype(float)/[self.discrete_phi-1,self.discrete_theta]				

		# SHIFTING BACK TO CANONICAL ACTIONS
		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP
		self.angular_action_size = 4
		self.angular_action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1]])
		# Negative theta, positive theta., negative phi, positive phi.

		# Defining transition model.
		self.trans_space = 3
		
		# self.trans = npy.zeros((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		# Defining angular transition models. 
		self.angular_trans = npy.ones((self.angular_action_size,self.trans_space,self.trans_space))
		
		for k in range(self.action_size):
			self.trans[k] /= self.trans[k].sum()

		# Normalizing angular transition models.
		for k in range(self.angular_action_size):			
			self.angular_trans[k] /= self.angular_trans[k].sum()

		self.action_counter = npy.zeros(self.action_size+self.angular_action_size)
		# Defining observation model.
		self.obs_space = 5
		
		# Discount
		self.gamma = 0.95

		# Defining belief variables.
		self.from_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.to_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.target_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.intermed_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.sensitivity = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))

		# Defining angular beliefs.
		self.from_angular_belief = npy.zeros((self.discrete_phi,self.discrete_theta))
		self.to_angular_belief = npy.zeros((self.discrete_phi,self.discrete_theta))
		self.target_angular_belief = npy.zeros((self.discrete_phi,self.discrete_theta))
		self.intermed_angular_belief = npy.zeros((self.discrete_phi,self.discrete_theta))
		self.sensitivity_angular_belief = npy.zeros((self.discrete_phi,self.discrete_theta))

		# Defining extended belief states. 
		self.w = self.trans_space/2
		self.to_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))
		self.from_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))

		# Extended angular belief states. 
		self.to_angular_ext = npy.zeros((self.discrete_phi+2*self.w,self.discrete_theta+2*self.w))
		self.from_angular_ext = npy.zeros((self.discrete_phi+2*self.w,self.discrete_theta+2*self.w))

		# Defining trajectory
		self.orig_traj = []
		self.orig_vel = []

		self.beta = npy.zeros(self.action_size)
		self.angular_beta = npy.zeros(self.angular_action_size)

		# Defining observation model related variables. 
		self.obs_space = 4
		self.obs_model = npy.zeros((self.obs_space,self.obs_space,self.obs_space))
		self.angular_obs_model = npy.zeros((self.obs_space,self.obs_space))

		self.h = self.obs_space-1
		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))
		self.extended_angular_obs_belief = npy.zeros((self.discrete_phi+self.h*2,self.discrete_theta+2*self.h))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 2
		self.annealing_rate = 0.1

		# Setting training parameters: 
		self.epochs = 5

	def load_trajectory(self, traj, actions, orientation, angular_vel):

		# Assume the trajectory file has positions and velocities.
		self.orig_traj = traj[0:len(traj):20,:]
		self.orig_vel = npy.diff(self.orig_traj,axis=0)
		self.orig_traj = self.orig_traj[:len(self.orig_vel),:]

		self.orig_orient = orientation[0:len(orientation):20,:]
		self.orig_angular_vel = npy.diff(self.orig_orient,axis=0)
		self.orig_orient = orientation[:len(self.orig_angular_vel),:]

		# self.orig_traj = traj
		# self.orig_vel = actions

		# self.orig_orient = orientation
		# self.orig_angular_vel = angular_vel

		# Linear trajectory interpolation and velocity interpolation array.
		self.interp_traj = npy.zeros((len(self.orig_traj),8,3),dtype='int')
		self.interp_traj_percent = npy.zeros((len(self.orig_traj),8))

		self.interp_vel = npy.zeros((len(self.orig_vel),3,3),dtype='int')
		self.interp_vel_percent = npy.zeros((len(self.orig_traj),3))
		
		# Angular trajectory interplation and velocity interpolation arrays.
		self.interp_angular_traj = npy.zeros((len(self.orig_traj),4,2),dtype='int')
		self.interp_angular_percent = npy.zeros((len(self.orig_traj),4))

		self.interp_angular_vel = npy.zeros((len(self.orig_vel),2,2),dtype='int')
		self.interp_angular_vel_percent = npy.zeros((len(self.orig_vel),2))

		self.preprocess_canonical()
		self.preprocess_angular()
		self.initialize_pointset()

	def initialize_pointset(self):

		self.pointset = npy.zeros((64,3))
		add = [-1,0,1,2]

		for i in range(4):
			for j in range(4):
				for k in range(4):
					self.pointset[16*i+4*j+k] = [add[i],add[j],add[k]]

		self.alter_point_set = (self.pointset*self.grid_cell_size).reshape(4,4,4,3)
		# self.pointset = (self.pointset+1).astype(int)

		self.angular_pointset = npy.zeros((16,2))

		for i in range(4):
			for j in range(4):
				self.angular_pointset[4*i+j] = [add[i],add[j]]
		
		self.alter_angular_pointset = (self.angular_pointset*self.angular_grid_cell_size).reshape(4,4,2)

	def construct_from_ext_state(self):

		w = self.w
		self.from_state_ext[w:self.discrete_x+w,w:self.discrete_y+w,w:self.discrete_z+w] = copy.deepcopy(self.from_state_belief)
		self.from_angular_ext[w:self.discrete_phi+w,w:self.discrete_theta+w] = copy.deepcopy(self.from_angular_belief)		

	def belief_prediction(self):
		# Implements the motion update of the Bayes Filter.

		w = self.w
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		# Linear then angular
		self.to_state_ext[:,:,:] = 0
		self.to_angular_ext[:,:] = 0	

		for k in range(self.action_size):
			self.to_state_ext += self.beta[k]*signal.convolve(self.from_state_ext,self.trans[k],'same')

		for k in range(self.angular_action_size):
			self.to_angular_ext += self.angular_beta[k]*signal.convolve2d(self.from_angular_ext,self.angular_trans[k],'same')			

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

			# Angular folding: This is probably the key to extending this to angular domains. 
			self.to_angular_ext[i+1,:] += self.to_angular_ext[i,:]
			self.to_angular_ext[i+1,:] = 0
			self.to_angular_ext[self.discrete_phi+2*w-i-2,:] += self.to_angular_ext[self.discrete_phi+2*w-i-1,:]
			self.to_angular_ext[self.discrete_phi+2*w-i-1,:] = 0

			# Now for theta dimension:
		left_theta = self.to_angular_ext[:,:w].copy()
		right_theta = self.to_angular_ext[:,-w:].copy()

		self.to_angular_ext[:,:w] = 0
		self.to_angular_ext[:,-w:] = 0
		self.to_angular_ext[:,w:2*w] += left_theta
		self.to_angular_ext[:,-2*w:w] += right_theta

		self.intermed_belief = copy.deepcopy(self.to_state_ext[w:dx+w,w:dy+w,w:dz+w])
		self.intermed_belief /= self.intermed_belief.sum()

		self.intermed_angular_belief = copy.deepcopy(self.to_angular_ext[w:self.discrete_phi+w,w:self.discrete_theta+w])
		self.intermed_angular_belief /= self.intermed_angular_belief.sum()

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

		self.extended_angular_obs_belief[:,:] = 0.
		self.extended_angular_obs_belief[h:self.discrete_phi+h,h:self.discrete_theta+h] = self.intermed_angular_belief
		self.extended_angular_obs_belief[h+angular_obs[0]-1:h+angular_obs[0]+3,h+angular_obs[1]-1:h+angular_obs[1]+3] = npy.multiply(self.extended_angular_obs_belief[h+angular_obs[0]-1:h+angular_obs[0]+3,h+angular_obs[1]-1:h+angular_obs[1]+3],self.angular_obs_model)

		# # Actually obs[0]-h:obs[0]+h, but with extended belief, we add another h:
		# self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)		

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h])
		self.to_state_belief /= self.to_state_belief.sum()

		self.to_angular_belief = copy.deepcopy(self.extended_angular_obs_belief[h:self.discrete_phi+h,h:self.discrete_theta+h])
		self.to_angular_belief /= self.to_angular_belief.sum()

	def compute_sensitivity(self):

		obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size).astype(int)
		angular_obs = npy.floor((self.angular_observed_state - self.ang_traj_lower)/self.angular_grid_cell_size).astype(int)
		
		h = self.h
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		intermediate_sensitivity = npy.zeros((self.discrete_x+2*h,self.discrete_y+2*h,self.discrete_z+2*h))
		intermediate_sensitivity[h:dx+h ,h:dy+h,h:dz+h] = self.target_belief - self.to_state_belief

		intermediate_angular_sensitivity = npy.zeros((self.discrete_phi+2*h,self.discrete_theta+2*h))
		intermediate_angular_sensitivity[h:self.discrete_phi+h, h:self.discrete_theta+h] = self.target_angular_belief - self.to_angular_belief

		# First linear sensitivities.
		self.sensitivity = npy.zeros((self.discrete_x+2*h,self.discrete_y+2*h,self.discrete_z+2*h))		
		 
		self.sensitivity[h+obs[0]-1:h+obs[0]+3,h+obs[1]-1:h+obs[1]+3,h+obs[2]-1:h+obs[2]+3] = npy.multiply(intermediate_sensitivity[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)
		self.sensitivity = self.sensitivity[h:dx+h,h:dy+h,h:dz+h]

		# Now angular sensitivities.
		self.sensitivity_angular_belief = npy.zeros((self.discrete_phi+2*h,self.discrete_theta+2*h))
		self.sensitivity_angular_belief[h+angular_obs[0]-1:h+angular_obs[0]+3,h+angular_obs[1]-1:h+angular_obs[1]+3] = npy.multiply(intermediate_angular_sensitivity[h+angular_obs[0]-1:h+angular_obs[0]+3,h+angular_obs[1]-1:h+angular_obs[1]+3], self.angular_obs_model)

		self.sensitivity_angular_belief = self.sensitivity_angular_belief[h:self.discrete_phi+h,h:self.discrete_theta+h]

	def backprop_convolution(self, num_epochs):

		self.compute_sensitivity()
		w = self.w
		
		# Set learning rate and lambda value.		
		self.time_count +=1
		alpha = self.learning_rate - self.annealing_rate*num_epochs

		# Calculate basic gradient update. 
		
		flip_from = npy.flip(npy.flip(npy.flip(self.from_state_ext,axis=0),axis=1),axis=2)
		grad_update = -2*signal.convolve(flip_from,self.sensitivity,'valid')
		# grad_update = -2*(self.beta.sum())*signal.convolve(flip_from,self.sensitivity,'valid')		

		flip_ang_from = npy.flip(npy.flip(self.from_angular_ext,axis=0),axis=1)	
		ang_grad_update = -2*signal.convolve2d(flip_ang_from, self.sensitivity_angular_belief,'valid')	

		t0 = npy.zeros((self.trans_space,self.trans_space,self.trans_space))
		t1 = npy.ones((self.trans_space,self.trans_space,self.trans_space))

		at0 = npy.zeros((self.trans_space,self.trans_space))
		at1 = npy.ones((self.trans_space,self.trans_space))

		for k in range(self.action_size):
			act_grad_update = self.beta[k]*(grad_update + self.lamda*(self.trans[k].sum() -1.))
			self.trans[k] = npy.maximum(t0,npy.minimum(t1,self.trans[k]-alpha*act_grad_update))

			# Is this needed? 
			self.trans[k] /= self.trans[k].sum()

		for k in range(self.angular_action_size):
			act_ang_grad_update = self.angular_beta[k]*(ang_grad_update+self.lamda*(self.angular_trans[k].sum()-1.))
			self.angular_trans[k] = npy.maximum(at0,npy.minimum(at1,self.angular_trans[k]-alpha*act_ang_grad_update))

	def recurrence(self):
		# With Teacher Forcing: Setting next belief to the previous target / ground truth.
		self.from_state_belief = copy.deepcopy(self.target_belief)
		self.from_angular_belief = copy.deepcopy(self.target_angular_belief)

	def _powerset(self, iterable):
		# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

		# Hello
	def angular_interpolate_coefficients(self, angular_state):

		base_indices = npy.floor((angular_state-self.ang_traj_lower)/self.angular_grid_cell_size)
		base_point = self.angular_grid_cell_size*npy.floor(angular_state/self.angular_grid_cell_size)

		base_lengths = angular_state - base_point
		bases = []

		for index_set in self._powerset(range(self.angular_dimensions)):
			index_set = set(index_set)
			volume = 1 
			
			index_to_add = base_indices.copy()

			for i in range(self.angular_dimensions):
				if i in index_set:
					side_length = base_lengths[i]			
					
					# Check if the dimension being added is yaw; if yes, take modulus.
					if i==1:
						index_to_add[i] = (index_to_add[i]+1)%(self.discrete_theta)
					else:
						index_to_add[i] += 1

				else:
					side_length = self.angular_grid_cell_size[i] - base_lengths[i]

				volume *= side_length / self.angular_grid_cell_size[i]

			bases.append((volume, index_to_add))			

		return bases

	def interpolate_coefficients(self, point, traj_or_action=1):
		# VARIABLE GRID SIZE ALONG DIFFERENT DIMENSIONS
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

	def map_double_to_angular_action(self, doublet):

		if doublet[0]==-1:
			return 0
		if doublet[0]==1:
			return 1
		if doublet[1]==-1:
			return 2
		if doublet[1]==1:
			return 3

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

	def preprocess_angular(self):
		print("Preprocessing Angular Data.")

		# Loads angles from -pi to pi.
		norm_vector = [0.5,npy.pi]

		self.orig_orient /= norm_vector

		vel_norm_vector = npy.max(abs(self.orig_angular_vel),axis=0)
		self.orig_angular_vel /= vel_norm_vector

		for t in range(len(self.orig_orient)-1):

			split = self.angular_interpolate_coefficients(self.orig_orient[t])
			count = 0

			for percent, indices in split:
				self.interp_angular_traj[t,count] = indices
				self.interp_angular_percent[t,count] = percent
				count +=1

			ang_vel = self.orig_angular_vel[t]

			self.interp_angular_vel[t,0] = [abs(ang_vel[0])/ang_vel[0],0]
			self.interp_angular_vel[t,1] = [0,abs(ang_vel[1])/ang_vel[1]]
			self.interp_angular_vel_percent[t] = abs(ang_vel)

	def preprocess_canonical(self):
		print("Preprocessing the Data.")

		# Normalize trajectory.
		norm_vector = [2.5,2.5,1.]
		# norm_vector = [1.,1.,1.]
		# norm_vector = npy.array([1.1,1.1,3.])
		# norm_vector = [3.,3.,3.]

		self.orig_traj /= norm_vector

		# Normalize actions (velocities).
		# self.orig_vel /= norm_vector

		# Currently only max norm per dimension.
		vel_norm_vector = npy.max(abs(self.orig_vel),axis=0)
		print(self.orig_vel)
		self.orig_vel /= vel_norm_vector

		for t in range(len(self.orig_traj)-1):
			
			# Trajectory. 
			split = self.interpolate_coefficients(self.orig_traj[t],1)
			count = 0
			for percent, indices in split: 
				self.interp_traj[t,count] = indices
				self.interp_traj_percent[t,count] = percent
				count += 1

			# Action. 
			# vel = self.orig_vel[t]/npy.linalg.norm(self.orig_vel[t])
			vel = self.orig_vel[t]


			self.interp_vel[t,0] = [abs(vel[0])/vel[0],0,0]
			self.interp_vel[t,1] = [0,abs(vel[1])/vel[1],0]
			self.interp_vel[t,2] = [0,0,abs(vel[2])/vel[2]]

			self.interp_vel_percent[t] = abs(vel)

		npy.save("Interp_Traj.npy",self.interp_traj)
		npy.save("Interp_Vel.npy",self.interp_vel)
		npy.save("Interp_Traj_Percent.npy",self.interp_traj_percent)
		npy.save("Interp_Vel_Percent.npy",self.interp_vel_percent)

	def parse_data(self, timepoint):
		# Preprocess data? 

		# Setting from state belief from interp_traj.
		# For each of the 8 grid points, set the value of belief = percent at that point. 
		# This should sum to 1.

		self.beta[:] = 0.
		self.angular_beta[:] = 0.

		self.target_belief[:,:,:] = 0.
		self.from_state_belief[:,:,:] = 0.

		self.target_angular_belief[:,:] = 0.
		self.from_angular_belief[:,:] = 0.

		for k in range(8):
			
			self.from_state_belief[self.interp_traj[timepoint,k,0],self.interp_traj[timepoint,k,1],self.interp_traj[timepoint,k,2]] = self.interp_traj_percent[timepoint,k]

			# Must also set the target belief. 
			self.target_belief[self.interp_traj[timepoint+1,k,0],self.interp_traj[timepoint+1,k,1],self.interp_traj[timepoint+1,k,2]] = self.interp_traj_percent[timepoint+1,k]

		# Now repeating tfor the angular.
		for k in range(4):
			self.from_angular_belief[self.interp_angular_traj[timepoint,k,0],self.interp_angular_traj[timepoint,k,1]] = self.interp_angular_percent[timepoint,k]
			# Setting target angular belief: 
			self.target_angular_belief[self.interp_angular_traj[timepoint+1,k,0],self.interp_angular_traj[timepoint+1,k,1]] = self.interp_angular_percent[timepoint+1,k]

		# Setting beta.
		# Map triplet indices to action index, set that value of beta to percent.
		# self.beta[self.map_triplet_to_action(self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2])] = self.interp_vel[timepoint,k,3] 	
		for k in range(3):
			self.beta[self.map_triplet_to_action_canonical([self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2]])] = self.interp_vel_percent[timepoint,k] 
		for k in range(2):
			self.angular_beta[self.map_double_to_angular_action([self.interp_angular_vel[timepoint,k,0],self.interp_angular_vel[timepoint,k,1]])] = self.interp_angular_vel_percent[timepoint,k]

		# This may be wrong; doesn't matter.
		# self.action_counter += [self.beta,self.angular_beta]
		self.action_counter[:self.action_size] += self.beta
		self.action_counter[self.action_size:] += self.angular_beta
		
		self.observed_state = self.orig_traj[timepoint]
		self.angular_observed_state = self.orig_orient[timepoint]

		mean = self.observed_state - self.grid_cell_size*npy.floor(self.observed_state/self.grid_cell_size)
		self.obs_model = mvn.pdf(self.alter_point_set,mean=mean,cov=0.005)
		self.obs_model /= self.obs_model.sum()

		angular_mean = self.angular_observed_state - self.angular_grid_cell_size*npy.floor(self.angular_observed_state/self.angular_grid_cell_size)
		self.angular_obs_model = mvn.pdf(self.alter_angular_pointset,mean=angular_mean,cov=0.005)
		self.angular_obs_model /= self.angular_obs_model.sum()

	def train_timepoint(self, timepoint, num_epochs):

		# Parse Data:
			# Set target belief, beta, etc...

		self.parse_data(timepoint)

		# Construct the from_extended_state for belief propagation
		self.construct_from_ext_state()
		
		# Propagate belief: 
			# Convolve with Trans model and merge intermediate beliefs.
		self.belief_prediction()
			# Correct Intermediate Belief (Observation Fusion)
		self.belief_correction()

		# Backpropagate with ground truth belief.
		self.backprop_convolution(num_epochs)

		# Recurrence. 
		self.recurrence()

	def train_BPRCNN(self):

		# Iterate over number of epochs.
		# Similar to QMDP Training paradigm.
		for i in range(self.epochs):

			print("Training Epoch: ",i)

			# Sequential training; later implement Shuffling / Experience Replay.
			
			# for j in range(len(self.traj)-1):
			for j in range(len(self.interp_traj)-2):
				print("Training epoch: {0} Time Step: {1}".format(i,j))
				self.train_timepoint(j,i)

			print("Saving the Model.")
			self.save_model()
			print(self.trans)

			print("ACTION COUNTER:", self.action_counter)

	def save_model(self):
		npy.save("Learnt_Transition_Linear.npy",self.trans)
		npy.save("Learnt_Transition_Angular.npy",self.angular_trans)

def main(args):    

	bprcnn = BPRCNN()

	# Load the CSV file, ignore first line.
	# traj_csv = npy.genfromtxt(str(sys.argv[1]),delimiter=',',usecols=[5,6,7,8,9,10,11,48,49,50,51,52,53])[1:]
	
	traj = npy.load(str(sys.argv[1]))
	actions = npy.load(str(sys.argv[2]))

	orient = npy.load(str(sys.argv[3]))
	angular_vel = npy.load(str(sys.argv[4]))

	# Pick up trajectories and linear velocities as actions.
	# bprcnn.load_trajectory(traj_csv[10000:,:3], traj_csv[10000:,7:10])

	# i = int(sys.argv[1])
	# FILE_DIR = "/home/tanmay/Research/Code/DeepVectorPolicyFields/Data/trajectories/T{0}".format(i)

	# traj = npy.load(os.path.join(FILE_DIR,"Downsamp_Actual_Traj.npy"))
	# actions = npy.load(os.path.join(FILE_DIR,"Commanded_Vel.npy"))

	# for i in range(9):

	bprcnn.load_trajectory(traj,actions,orient,angular_vel)
	bprcnn.train_BPRCNN()

if __name__ == '__main__':
	main(sys.argv)
