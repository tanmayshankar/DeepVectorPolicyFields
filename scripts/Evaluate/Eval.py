#!/usr/bin/env python
from headers import *

def main(args):    

	FILE_DIR = "/home/tanmay/Research/Code/VectorFields/Data/Quad/T{0}"

	action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
	act = action_space+1

	expected_vel = npy.zeros((10,6,3))
	for p in range(10):

		trans = npy.load(os.path.join(FILE_DIR.format(p+1),"Learnt_Transition.npy"))
		
		for q in range(6):
			
			for i in range(3):
				for j in range(3):
					for k in range(3):
						expected_vel[p,q] += trans[q,i,j,k]*npy.array([i-1,j-1,k-1])

			expected_vel[p,q] /= npy.linalg.norm(expected_vel[p,q])

	npy.save("Expected_Vel.npy",expected_vel)

	# Calculate Cosine Similarities
	cosines = npy.zeros((10,6))

	for p in range(10):
		for q in range(6):
			cosines[p,q]=npy.dot(action_space[q],expected_vel[p,q])
		print("Trajectory: ", p)
		print(cosines[p])

	npy.save("Cosine_Similarities.npy",cosines)

if __name__ == '__main__':
	main(sys.argv)

