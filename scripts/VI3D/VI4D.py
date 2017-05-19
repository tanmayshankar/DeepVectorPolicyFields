#!/usr/bin/env python 
from headers import *

class VI_RCNN():

	def __init__(self): 

		self.discrete_x = 50
		self.discrete_y = 50
		self.discrete_z = 32
		self.discrete_yaw = 36

		self.dimensions = 3
		self.action_size = 6

		# SHIFTING BACK TO CANONICAL ACTIONS
		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		# Defining transition model.
		self.trans_space = 3
		# self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space, self.trans_space))

		# for k in range(self.action_size):
		# 	self.trans[k] /= self.trans[k].sum()

		self.action_counter = npy.zeros(self.action_size)
		
		# Defining Value function, policy, etc. 	
		# Now the value function and the policy etc. must be a function of XYZ and Yaw.
		self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z, self.discrete_yaw))
		self.policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_yaw),dtype=int)
		# self.reward = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_yaw))
		self.Qvalues = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_yaw))

		# Discount
		self.gamma = 0.98
		self.beta = npy.zeros(self.action_size)

		# Setting number of iterations
		self.iterations = 200

	def load_model(self, reward, trans):
		# Loading reward and transition. 
		self.reward = reward
		# self.trans = npy.flip(npy.flip(npy.flip(trans,axis=1),axis=2),axis=3)
		self.trans = trans

		# for k in range(self.action_size):
		# 	self.trans[k] = npy.flip(npy.flip(npy.flip(self.trans[k],axis=0),axis=1),axis=2)

		for k in range(self.action_size):
			self.trans[k] = npy.flip(npy.flip(npy.flip(npy.flip(self.trans[k],axis=0),axis=1),axis=2),axis=3)
		
	def conv_layer(self):
		# Convolutional Layer
		for k in range(self.action_size):
			self.Qvalues[k] = signal.convolve(self.value_function,self.trans[k],'same')

		# Now modifying convolutional layer of the VI RCNN to take into account that yaw wraps around. 

		# Construct extended value function - must extended along every dimension, then implement as a 4D Valid conv, instead of Same conv.
		w = 1
		self.extended_value = npy.zeros((self.discrete_x+2*w, self.discrete_y+2*w, self.discrete_z+2*w, self.discrete_yaw+2*w))
		self.extended_value[:,:,:,w:-w-1] = self.value_function
		
		self.extended_value[:,:,:,0] = self.value_function[:,:,:,-1]
		self.extended_value[:,:,:,-1] = self.value_function[:,:,:,0]	

		# Convolve.
		for k in range(self.action_size):
			self.Qvalues[k] = signal.convolve(self.extended_value,self.trans[k],'valid')

		print(self.Qvalues.shape)

	def max_pool(self):
		# Pooling layer across action channel.
		self.value_function = npy.amax(self.Qvalues,axis=0)
		self.policy = npy.argmax(self.Qvalues,axis=0)

	def reward_bias(self):
		# Adding fixed bias.
		self.Qvalues = self.gamma*self.Qvalues + self.reward

	def bound_policy(self):
		# Ensuring the policy doesn't try to go out of bounds. 
		self.policy[0,:,:] = 1
		self.policy[self.discrete_x-1,:,:] = 0
		self.policy[:,0,:] = 3
		self.policy[:,self.discrete_y-1,:] = 2
		self.policy[:,:,0] = 5
		self.policy[:,:,self.discrete_z-1] = 4

	def recurrent_value_iteration(self):
		# Call Value Iteration.
		for i in range(self.iterations):
			print("Iteration: ",i)
			self.conv_layer()
			self.reward_bias()
			self.max_pool()
			

		self.soft_continuous_policy()

	def soft_continuous_policy(self):

		self.continuous_policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z, self.discrete_yaw, self.dimensions))
		self.softmax = npy.zeros((self.action_size,self.discrete_x,self.discrete_y,self.discrete_z, self.discrete_yaw))

		# for k in range(self.action_size):
		self.softmax = npy.exp(self.Qvalues)/(npy.sum(npy.exp(self.Qvalues),axis=0))
	
		# self.continuous_policy += self.softmax[k]
		for i in range(self.discrete_x):
			for j in range(self.discrete_y):
				for k in range(self.discrete_z):
					for y in range(self.discrete_yaw):
						for act in range(self.action_size):

							self.continuous_policy[i,j,k,y] += self.softmax[act,i,j,k,y]*self.action_space[act]

	def save_model(self):
		npy.save("Planned_Policy.npy",self.policy)
		npy.save("Planned_Value.npy",self.value_function)
		npy.save("Planned_QValue.npy",self.Qvalues)
		npy.save("Continuous_Policy.npy",self.continuous_policy)

def main(args):    

	vircnn = VI_RCNN()

	reward = npy.load(str(sys.argv[1]))
	trans = npy.load(str(sys.argv[2]))

	vircnn.load_model(reward,trans)
	vircnn.recurrent_value_iteration()
	vircnn.save_model()

if __name__ == '__main__':
	main(sys.argv)




