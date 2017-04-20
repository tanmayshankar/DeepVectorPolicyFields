#!/usr/bin/env python 
from headers import *

class QMDP_RCNN():

	def __init__(self):

		self.discrete_x = 51
		self.discrete_y = 51
		self.discrete_z = 11

		self.dimensions = 3
		self.action_size = 6

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

		# Defining transition model.
		self.trans_space = 3
		# self.trans = npy.random.random((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		# self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		# for k in range(self.action_size):
		# 	self.trans[k] /= self.trans[k].sum()

		self.action_counter = npy.zeros(self.action_size)

		# Defining observation model.
		self.obs_space = 5
		
		# Defining Value function, policy, etc. 	
		self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z))
		self.policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z))
		self.reward = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))
		self.Qvalues = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))

		# Defining belief space Q values; softmax values. 
		self.belief_space_q = npy.zeros(self.action_size)
		self.softmax_q = npy.zeros(self.action_size)

		# Discount
		self.gamma = 0.95

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
		# self.h = self.obs_space/2 
		self.h = self.obs_space-1

		self.extended_obs_belief = npy.zeros((self.discrete_x+self.h*2,self.discrete_y+self.h*2,self.discrete_z+self.h*2))

		# Setting hyperparameters
		self.time_count = 0
		self.lamda = 1
		self.learning_rate = 0.5
		self.annealing_rate = 0.02

		# Setting training parameters: 
		self.epochs = 5

	def load_trajectory(self, traj, actions):

		# Assume the trajectory file has positions and velocities.
		self.orig_traj = traj[0:len(traj):5,:]
		self.orig_vel = actions[0:len(traj):5,:]

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

	# VARIABLE GRID SIZE ALONG DIFFERENT DIMENSIONS:
	def interpolate_coefficients(self, point, traj_or_action=1):

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
		norm_vector = [2.5,2.5,1.]		
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

	def parse_data(self, timepoint):
		# Preprocess data? 

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

	def recurrence(self):
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

		# Update Q(b(s),a) values by fusing Q(s,a) and b(s) as per the QMDP approximation.
		self.update_QMDP_values()	

		# Update "choices" of action (corresponding to predicted beta values) as the softmax of Q(b(s),a).
		self.calc_softmax_beta()

		alpha = self.learning_rate - self.time_count*self.annealing_rate

		# Remember, now Target Actions are not a one hot encoding; rather they are represented by beta.
		for k in range(self.action_size):
			self.reward[k] -= alpha * (self.softmax_q[k] - self.beta[k])*self.from_state_belief

	def max_pool(self):
		# Pooling along action channel.
		self.value_function = npy.amax(self.Qvalues,axis=0)
		self.policy = npy.argmax(self.Qvalues,axis=0)

	def conv_layer(self):
		# Convolving current value estimate with transition filters.

		# Kernel flipping? 
		trans_flip = npy.flip(npy.flip(npy.flip(self.trans,axis=1),axis=2),axis=3)

		for k in range(self.action_size):
			self.Qvalues[k] = signal.convolve(self.value_function, trans_flip[k], 'same')

	def feedback(self):
		self.max_pool()
		self.conv_layer()

	def train_timepoint(self, timepoint, num_epochs):

		# Parse Data:
			# Set target belief, beta, etc...
		self.parse_data(timepoint)

		# Construct the from_extended_state for belief propagation.
		self.construct_from_ext_state()
		
		# Propagate belief: 
		# Convolve with Trans model and merge intermediate beliefs.
		self.belief_prediction()
		# Correct Intermediate Belief (Observation Fusion)
		self.belief_correction()

		# Backpropagate the Cross Entropy / Negative Log Likelihood. 
		# Equivalent to the KL Divergence; since the target distribution is fixed.
		self.backprop_reward(num_epochs)

		# Update Q Values: This is different from Feedback
		self.update_Q_estimate(0.99)

		# Recurrence. 
		self.recurrence()

		# Feedback - call in train_QMDPRCNN; after trajectories and epochs.

	def train_QMDPRCNN(self):

		# Training without experience Replay for now. 
		for e in range(self.epochs):

			print("Training Epoch: ",e)

			# For each transition in the trajectory.
			for j in range(len(self.interp_traj)-1):

				print("Training: Epoch: {0}, Time Step: {1}".format(e,j))
				self.train_timepoint(j,e)
 
 			self.feedback()
			self.save_model()

			self.time_count +=1

	def save_model(self):

		# Saving reward, value, q, and policy.
		npy.save("Learnt_Reward.npy",self.reward)
		npy.save("Learnt_Policy.npy",self.policy)
		npy.save("Learnt_Value.npy",self.value_function)
		npy.save("Learnt_QValue.npy",self.Qvalues)

def main(args):

	qmdprcnn = QMDP_RCNN()

	# traj = npy.load(str(sys.argv[1]))
	# print(traj)
	# actions = npy.load(str(sys.argv[2]))


	trans = npy.load(str(sys.argv[1]))

	a = [2,3,5,6]

	FILE_DIR = "/home/tanmay/Research/Code/DeepVectorPolicyFields/Data/trajectories/T{0}"

	qmdprcnn.load_transition(trans)

	for i in a:
		traj = npy.load(os.path.join(FILE_DIR.format(i),"Downsamp_Actual_Traj.npy"))
		actions = npy.load(os.path.join(FILE_DIR.format(i),"Commanded_Vel.npy"))

		qmdprcnn.load_trajectory(traj,actions)
		qmdprcnn.train_QMDPRCNN()

if __name__ == '__main__':
	main(sys.argv)