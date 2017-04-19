#!/usr/bin/env python 
from headers import *

class VI_RCNN():

	def __init__(self):

		self.discrete_x = 51
		self.discrete_y = 51
		self.discrete_z = 11

		self.dimensions = 3
		self.action_size = 6

		# SHIFTING BACK TO CANONICAL ACTIONS
		self.action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
		# ACTIONS: LEFT, RIGHT, BACKWARD, FRONT, DOWN, UP

		# Defining transition model.
		self.trans_space = 3
		self.trans = npy.ones((self.action_size,self.trans_space,self.trans_space, self.trans_space))
		# for k in range(self.action_size):
		# 	self.trans[k] /= self.trans[k].sum()

		self.action_counter = npy.zeros(self.action_size)
		
		# Defining Value function, policy, etc. 	
		self.value_function = npy.zeros((self.discrete_x, self.discrete_y, self.discrete_z))
		self.policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z),dtype=int)
		self.reward = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))
		self.Qvalues = npy.zeros((self.action_size,self.discrete_x,self.discrete_y, self.discrete_z))

		# Discount
		self.gamma = 0.95
		self.beta = npy.zeros(self.action_size)

		# Setting number of iterations
		self.iterations = 100

	def load_model(self, reward, trans):
		# Loading reward and transition. 
		self.reward = reward
		# self.trans = npy.flip(npy.flip(npy.flip(trans,axis=1),axis=2),axis=3)
		self.trans = trans

		for k in range(self.action_size):
			self.trans[k] = npy.flip(npy.flip(npy.flip(self.trans[k],axis=0),axis=1),axis=2)
		
	def conv_layer(self):
		# Convolutional Layer
		for k in range(self.action_size):
			self.Qvalues[k] = signal.convolve(self.value_function,self.trans[k],'same')

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
			self.conv_layer()
			self.reward_bias()
			self.max_pool()
			print("Iteration: ",i)

		self.soft_continuous_policy()

	def soft_continuous_policy(self):

		self.continuous_policy = npy.zeros((self.discrete_x,self.discrete_y, self.discrete_z,self.dimensions))
		self.softmax = npy.zeros((self.action_size,self.discrete_x,self.discrete_y,self.discrete_z))

		# for k in range(self.action_size):
		self.softmax = npy.exp(self.Qvalues)/(npy.sum(npy.exp(self.Qvalues),axis=0))
	
		# self.continuous_policy += self.softmax[k]
		for i in range(self.discrete_x):
			for j in range(self.discrete_y):
				for k in range(self.discrete_z):
					for act in range(self.action_size):

						self.continuous_policy[i,j,k] += self.softmax[act,i,j,k]*self.action_space[act]

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




