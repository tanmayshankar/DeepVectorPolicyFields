#!/usr/bin/env python
from headers import *
from variables import *

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
		self.action_cell = 1
		# Assuming lower is along all dimensions. 
		self.traj_lower = -1
		self.traj_cell = 0.1

		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		# Defining transition model.
		self.trans_space = 3
		self.trans = npy.random.random((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		
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

		# Defining extended belief states. 
		self.w = self.trans_space/2
		self.to_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))
		self.from_state_ext = npy.zeros((self.discrete_x+2*self.w,self.discrete_y+2*self.w,self.discrete_z+2*self.w))

		# Defining trajectory
		self.traj = []
		self.actions = []
		self.beta = npy.zeros(self.action_size)

		# Defining observation model related variables. 
		self.obs_space = 3
		self.obs_model = npy.zeros((self.obs_space,self.obs_space,self.obs_space))
		self.h = self.obs_space/2
		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 0

		# Setting training parameters: 
		self.epochs = 1

	def load_trajectory(self, traj, actions):

		# Assume the trajectory file has positions and velocities.
		self.traj = traj
		self.actions = actions

	def construct_from_ext_state(self):

		w = self.w
		d = self.discrete_x
		# self.from_state_ext[w:d+w,w:d+w,w:d+w] = copy.deepcopy(self.from_state_ext)
		self.from_state_ext[w:self.discrete_x+w,w:self.discrete_y+w,w:self.discrete_z+w] = copy.deepcopy(self.from_state_ext)

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
		dx = self.discrete_x
		dy = self.discrete_y
		dz = self.discrete_z

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
		h = self.h
		obs = self.observed_state
		
		self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h] = self.intermed_belief
		self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h] = npy.multiply(self.extended_obs_belief[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)

		self.to_state_belief = copy.deepcopy(self.extended_obs_belief[h:dx+h,h:dy+h,h:dz+h])
		self.to_state_belief /= self.to_state_belief.sum()

	def compute_sensititives(self):

		# Compute the sensitivity values. 
		self.sensitivity = self.target_belief - self.from_state_belief

		obs = self.observed_state
		h = self.h

		# Redefining the sensitivity values to be (target_b - pred_b)*obs_model = F.
		self.sensitivity = npy.multiply(self.sensitivity[obs[0]:obs[0]+2*h,obs[1]:obs[1]+2*h,obs[2]:obs[2]+2*h],self.obs_model)

	def backprop_convolution(self):

		self.compute_sensiti
		tives()
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

	def _powerset(self, iterable):
		# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

	def interpolate_coefficients(self, point, traj_or_action=1):
	# def interpolate_coefficients(self, point):

		# Choose whether we are interpolating a trajectory or an action data point.
		# If traj_or_action is 0, it's an action, if 1, it's a trajectory.
		# lower = traj_or_action * self.traj_lower + (1-traj_or_action)*self.action_lower

		# Now lower is just uniformly -1. 
		lower = -1
		grid_cell_size = traj_or_action * self.traj_cell + (1-traj_or_action)*self.action_cell

		base_indices = npy.floor((point-lower)/grid_cell_size)
		base_point = sgrid_cell_size*npy.floor(point/grid_cell_size)		
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
					# point_to_add += self.grid_cell_size
					index_to_add[i] += 1
				else:
					side_length = self.grid_cell_size - base_lengths[i]

				volume *= side_length / grid_cell_size

			# bases.append((volume, point_to_add, index_to_add))			
			bases.append((volume, index_to_add))			

		return bases

	# def interpolate_coefficients(self, point, traj_or_action=1):
	# # def interpolate_coefficients(self, point):

	# 	# Choose whether we are interpolating a trajectory or an action data point.
	# 	# If traj_or_action is 0, it's an action, if 1, it's a trajectory.
	# 	lower = traj_or_action * self.traj_lower + (1-traj_or_action)*self.action_lower
	# 	grid_cell_size = traj_or_action * self.traj_cell + (1-traj_or_action)*1

	# 	base_indices = npy.floor((point-self.lower)/self.grid_cell_size)
	# 	base_point = self.grid_cell_size*npy.floor(point/self.grid_cell_size)		
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
	# 				side_length = self.grid_cell_size - base_lengths[i]

	# 			volume *= side_length / self.grid_cell_size

	# 		# bases.append((volume, point_to_add, index_to_add))			
	# 		bases.append((volume, index_to_add))			

	# 	return bases

	def map_triplet_to_action(self, triplet):

		# Assuming the triplets are indices, not the coordinates of the point
		# at which the action is being interpolated.
		return (triplet[0]+triplet[1]*3+triplet[2]*9)

	def preprocess_trajectory(self):

		# Normalize trajectory.
		norm_vector = [2.5,2.5,1.]
		self.traj /= norm_vector

		# Normalize actions (velocities).
		vel_norm_vector = npy.max(abs(self.actions),axis=0)
		self.actions /= vel_norm_vector

		for t in range(len(self.traj)):
			
			# Trajectory. 
			self.interpolate_coefficients(self.traj[t],1)

			# Action. 


			self.interpolate_coefficients(self.actions[t]/npy.linalg.norm(self.actions[t]),0)


	def parse_data(self):
		# Preprocess data? 



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
			# Sequential training; later implement Shuffling / Experience Replay.
			for j in range(len(self.traj)):
				self.train_timepoint(j)

def main(args):    

	bprcnn = BPRCNN()

	# Load the CSV file, ignore first line.
	traj_csv = npy.genfromtxt(str(sys.argv[1]),delimiter=',',usecols=[5,6,7,8,9,10,11,48,49,50,51,52,53])[1:]
	
	# Pick up trajectories and linear velocities as actions.
	bprcnn.load_trajectory(traj_csv[:,:3], traj_cs[:,7:10])
	bprcnn.preprocess_trajectory()

	bprcnn.train_BPRCNN()

if __name__ == '__main__':
    main(sys.argv)




