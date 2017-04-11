#!/usr/bin/env python
from headers import *
# from variables import *

class BPRCNN():

	def __init__(self):

		# Defining common variables.
		self.discrete = 50
		self.discrete_x = 50
		self.discrete_y = 50
		self.discrete_z = 10

		self.dimensions = 3
		# self.action_size = 6
		self.action_size = 27

		# Setting discretization variables
		self.action_lower = -1
		self.action_upper = +1
		self.action_cell = npy.ones(self.dimensions)

		# Assuming lower is along all dimensions. 		
		self.traj_lower = npy.array([-1,-1,0])
		self.traj_upper = +1
		# self.traj_cell = 0.1
		# self.traj_cell = (self.traj_upper-self.traj_lower)/self.

		self.grid_cell_size = npy.array((self.traj_upper-self.traj_lower)).astype(float)/[self.discrete_x, self.discrete_y, self.discrete_z]		
		# self.grid_cell_size[2] = 1./self.discrete_z

		# self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		self.action_space = npy.array([ [-1., -1., -1.],
										[ 0., -1., -1.],
										[ 1., -1., -1.],
										[-1.,  0., -1.],
										[ 0.,  0., -1.],
										[ 1.,  0., -1.],
										[-1.,  1., -1.],
										[ 0.,  1., -1.],
										[ 1.,  1., -1.],
										[-1., -1.,  0.],
										[ 0., -1.,  0.],
										[ 1., -1.,  0.],
										[-1.,  0.,  0.],
										[ 0.,  0.,  0.],
										[ 1.,  0.,  0.],
										[-1.,  1.,  0.],
										[ 0.,  1.,  0.],
										[ 1.,  1.,  0.],
										[-1., -1.,  1.],
										[ 0., -1.,  1.],
										[ 1., -1.,  1.],
										[-1.,  0.,  1.],
										[ 0.,  0.,  1.],
										[ 1.,  0.,  1.],
										[-1.,  1.,  1.],
										[ 0.,  1.,  1.],
										[ 1.,  1.,  1.]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		# Defining transition model.
		self.trans_space = 3
		self.trans = npy.random.random((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		for k in range(self.action_size):
			self.trans[k] /= self.trans[k].sum()
		
		# Defining observation model.
		self.obs_space = 5
		
		# Defining Value function, policy, etc. 	
		self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z))
		self.policy = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z))

		# Discount
		self.gamma = 0.95

		# Defining belief variables.
		self.from_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.to_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.target_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		# self.corrected_to_state_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.intermed_belief = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))
		self.sensitivity = npy.zeros((self.discrete_x,self.discrete_y,self.discrete_z))

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
		# self.h = self.obs_space/2 

		self.h = self.obs_space-1
		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 0.05
		self.annealing_rate = 0.

		# Setting training parameters: 
		self.epochs = 1

	def load_trajectory(self, traj, actions):

		# Assume the trajectory file has positions and velocities.
		self.orig_traj = traj[0:len(traj):20,:]
		self.orig_vel = actions[0:len(traj):20,:]

		self.orig_vel = npy.diff(self.orig_traj,axis=0)
		self.orig_traj = self.orig_traj[:len(self.orig_vel),:]

		# self.orig_traj = traj
		# self.orig_vel = actions

		self.interp_traj = npy.zeros((len(self.orig_traj),8,3),dtype='int')
		self.interp_vel = npy.zeros((len(self.orig_traj),8,3),dtype='int')

		self.interp_traj_percent = npy.zeros((len(self.orig_traj),8))
		self.interp_vel_percent = npy.zeros((len(self.orig_traj),8))
		
		self.preprocess_trajectory()
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
		self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h] = self.intermed_belief		
		self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3] = npy.multiply(self.extended_obs_belief[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)

		# # Actually obs[0]-h:obs[0]+h, but with extended belief, we add another h:
		# self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)		

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h])
		self.to_state_belief /= self.to_state_belief.sum()

	def compute_sensitivities(self):

		# Compute the sensitivity values. 
		self.sensitivity = self.target_belief - self.from_state_belief

		obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size).astype(int)

		# obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size)
		h = self.h
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		# Redefining the sensitivity values to be (target_b - pred_b)*obs_model = F.
		# self.sensitivity = npy.multiply(self.sensitivity[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)

		# UPDATING TO THE GAUSSIAN KERNEL OBSERVATION MODEL AGAIN:
		intermediate_sensitivity = npy.zeros((self.discrete_x+2*h,self.discrete_y+2*h,self.discrete_z+2*h))
		intermediate_sensitivity[h:dx+h,h:dy+h,h:dz+h] = self.sensitivity.copy()		
		self.sensitivity = npy.multiply(intermediate_sensitivity[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)		

	def compute_sensitivity(self):

		obs = npy.floor((self.observed_state - self.traj_lower)/self.grid_cell_size).astype(int)
		h = self.h
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

		intermediate_sensitivity = npy.zeros((self.discrete_x+2*h,self.discrete_y+2*h,self.discrete_z+2*h))
		intermediate_sensitivity[h:dx+h ,h:dy+h,h:dz+h] = self.target_belief - self.from_state_belief

		# self.sensitivity[:,:,:] = 0
		self.sensitivity = npy.zeros((self.discrete_x+2*h,self.discrete_y+2*h,self.discrete_z+2*h))		
		 
		self.sensitivity[h+obs[0]-1:h+obs[0]+3,h+obs[1]-1:h+obs[1]+3,h+obs[2]-1:h+obs[2]+3] = npy.multiply(intermediate_sensitivity[h+obs[0]-1:h+obs[0]+3, h+obs[1]-1:h+obs[1]+3, h+obs[2]-1:h+obs[2]+3], self.obs_model)
		self.sensitivity = self.sensitivity[h:dx+h,h:dy+h,h:dz+h]

	def backprop_convolution(self):

		# self.compute_sensitivities()
		self.compute_sensitivity()
		w = self.w
		
		# Set learning rate and lambda value.
		
		self.time_count +=1
		alpha = self.learning_rate - self.annealing_rate*self.time_count

		# Calculate basic gradient update. 
		grad_update = -2*signal.convolve(self.from_state_ext,self.sensitivity,'valid')		

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

	def _powerset(self, iterable):
		# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# VARIABLE GRID SIZE ALONG DIFFERENT DIMENSIONS:
	def interpolate_coefficients(self, point, traj_or_action=1):
	# def interpolate_coefficients(self, point):

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


# CONSTANT GRID SIZE ALONG ALL DIMENSIONS:
	# def interpolate_coefficients(self, point, traj_or_action=1):
	# # def interpolate_coefficients(self, point):

	# 	# Choose whether we are interpolating a trajectory or an action data point.

	# 	# Now lower is just uniformly -1. 
	# 	lower = -npy.ones(3)

	# 	# If traj_or_action is 0, it's an action, if 1, it's a trajectory.
	# 	# If trajectory, z lower must be 0.
	# 	lower[2] += traj_or_action

	# 	grid_cell_size = traj_or_action * self.traj_cell + (1-traj_or_action)*self.action_cell
		

	# 	base_indices = npy.floor((point-lower)/grid_cell_size)
	# 	base_point = grid_cell_size*npy.floor(point/grid_cell_size)		
	# 	base_lengths = point - base_point
	# 	bases = []

	# 	for index_set in self._powerset(range(self.dimensions)):
	# 		index_set = set(index_set)
	# 		volume = 1 
	# 		# point_to_add = base_point.copy()
	# 		index_to_add = base_indices.copy()

	# 		for i in range(self.dimensions):
	# 			if i in index_set:
	# 				side_length = base_lengths[i]			
	# 				# point_to_add += self.grid_cell_size
	# 				index_to_add[i] += 1
	# 			else:
	# 				side_length = grid_cell_size - base_lengths[i]

	# 			volume *= side_length / grid_cell_size

	# 		# bases.append((volume, point_to_add, index_to_add))			
	# 		bases.append((volume, index_to_add))			

	# 	return bases

	def map_triplet_to_action(self, triplet):

		# Assuming the triplets are indices, not the coordinates of the point
		# at which the action is being interpolated.
		return (triplet[0]+triplet[1]*3+triplet[2]*9)
		# return (triplet[0]*9+triplet[1]*3+triplet[2])

	def preprocess_trajectory(self):

		print("Preprocessing the Data.")

		# Normalize trajectory.
		# norm_vector = [2.5,2.5,1.]
		norm_vector = [3.,3.,3.]
		self.orig_traj /= norm_vector

		# Normalize actions (velocities).
		vel_norm_vector = npy.max(abs(self.orig_vel),axis=0)
		self.orig_vel /= vel_norm_vector

		# for t in range(len(self.traj)):
		for t in range(len(self.orig_traj)):
			
			# Trajectory. 
			split = self.interpolate_coefficients(self.orig_traj[t],1)
			count = 0
			for percent, indices in split: 
				self.interp_traj[t,count] = indices
				self.interp_traj_percent[t,count] = percent
				count += 1

			# Action. 
			# split = self.interpolate_coefficients(self.actions[t]/npy.linalg.norm(self.actions[t]),0)
			split = self.interpolate_coefficients(self.orig_vel[t]/npy.linalg.norm(self.orig_vel[t]),0)
			count = 0
			for percent, indices in split:
				self.interp_vel[t,count] = indices
				self.interp_vel_percent[t,count] = percent
				count += 1

		npy.save("Interp_Traj.npy",self.interp_traj)
		npy.save("Interp_Vel.npy",self.interp_vel)
		npy.save("Interp_Traj_Percent.npy",self.interp_traj_percent)
		npy.save("Interp_Vel_Percent.npy",self.interp_vel_percent)

	def parse_data(self, timepoint):
		# Preprocess data? 

		# Setting from state belief from interp_traj.
		# For each of the 8 grid points, set the value of belief = percent at that point. 
		# This should sum to 1.
		for k in range(8):
			
			self.from_state_belief[self.interp_traj[timepoint,k,0],self.interp_traj[timepoint,k,1],self.interp_traj[timepoint,k,2]] = self.interp_traj_percent[timepoint,k]

			# Setting beta.
			# Map triplet indices to action index, set that value of beta to percent.
			# self.beta[self.map_triplet_to_action(self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2])] = self.interp_vel[timepoint,k,3] 
			self.beta[self.map_triplet_to_action([self.interp_vel[timepoint,k,0],self.interp_vel[timepoint,k,1],self.interp_vel[timepoint,k,2]])] = self.interp_vel_percent[timepoint,k] 

		# Must also set the target belief. 
			self.target_belief[self.interp_traj[timepoint+1,k,0],self.interp_traj[timepoint+1,k,1],self.interp_traj[timepoint+1,k,2]] = self.interp_traj_percent[timepoint+1,k]

		self.observed_state = self.orig_traj[timepoint]
		mean = self.observed_state - self.grid_cell_size*npy.floor(self.observed_state/self.grid_cell_size)
		self.obs_model = mvn.pdf(self.alter_point_set,mean=mean,cov=0.001)
		self.obs_model /= self.obs_model.sum()

	def train_timepoint(self, timepoint):

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
		self.backprop_convolution()

		# Recurrence. 
		self.recurrence()

	def train_BPRCNN(self):

		# Iterate over number of epochs.
		# Similar to QMDP Training paradigm.
		for i in range(self.epochs):

			print("Training Epoch: ",i)

			# Sequential training; later implement Shuffling / Experience Replay.
			# for j in range(len(self.traj)-1):
			for j in range(len(self.interp_traj)-1):
				print("Training Time Step:",j)
				self.train_timepoint(j)

			print("Saving the Model.")
			self.save_model()

	def save_model(self):
		npy.save("Learnt_Transition.npy",self.trans)

def main(args):    

	bprcnn = BPRCNN()

	# Load the CSV file, ignore first line.
	traj_csv = npy.genfromtxt(str(sys.argv[1]),delimiter=',',usecols=[5,6,7,8,9,10,11,48,49,50,51,52,53])[1:]
	
	# Pick up trajectories and linear velocities as actions.
	bprcnn.load_trajectory(traj_csv[:,:3], traj_csv[:,7:10])
	bprcnn.train_BPRCNN()

if __name__ == '__main__':
	main(sys.argv)




