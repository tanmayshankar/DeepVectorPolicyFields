#!/usr/bin/env python
from headers import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main(args):    

	FILE_DIR = "/home/tanmay/Research/Code/VectorFields/Data/Quad/T{0}"

	action_space = npy.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
	act = action_space+1

	expected_vel = npy.zeros((10,6,3))
	covariance = npy.zeros((10,6,3))
	covariance_full = npy.zeros((10,6,3,3))

	for p in range(10):

		trans = npy.load(os.path.join(FILE_DIR.format(p+1),"Learnt_Transition.npy"))
		
		for q in range(6):
			
			for i in range(3):
				for j in range(3):
					for k in range(3):
						expected_vel[p,q] += trans[q,i,j,k]*npy.array([i-1,j-1,k-1])

		for q in range(6):

			for i in range(3):
				for j in range(3):
					for k in range(3):
						# Assuming diagonal:
						vect = npy.array([i-1,j-1,k-1]-expected_vel[p,q])
						covariance[p,q] += trans[q,i,j,k]*(vect**2)
						covariance_full[p,q] += trans[q,i,j,k]*npy.outer(vect,vect)

			# NOW ONLY NORMALIZING AFTER COMPUTING COVARIANCE
			# covariance[p,q] /= npy.linalg.norm(expected_vel[p,q]) #?

			# expected_vel[p,q] /= npy.linalg.norm(expected_vel[p,q])


	npy.save("Unnorm_Expected_Vel.npy",expected_vel)
	npy.save("Unnorm_Diag_Covariance.npy",covariance)
	npy.save("Unnorm_Full_Covariance.npy",covariance_full)

	for p in range(10):
		for q in range(6):

			covariance[p,q] /= npy.linalg.norm(expected_vel[p,q])
			expected_vel[p,q] /= npy.linalg.norm(expected_vel[p,q])

	npy.save("Norm_Expected_Vel.npy",expected_vel)
	npy.save("Norm_Diag_Covariance.npy",covariance)
	npy.save("Norm_Full_Covariance.npy",covariance_full)


	# Calculate Cosine Similarities
	cosines = npy.zeros((10,6))

	for p in range(10):
		for q in range(6):
			cosines[p,q]=npy.dot(action_space[q],expected_vel[p,q])
		print("Trajectory: ", p)
		print(cosines[p])

	npy.save("Cosine_Similarities.npy",cosines)

	acts = action_space.reshape(3,2,3)

	u = npy.linspace(0,2*npy.pi,100)
	v = npy.linspace(0,npy.pi,100)

	for p in range(10):
		for q in range(6):
			
			fig = plt.figure()
			ax = fig.gca(projection='3d')

			for k in range(3):
				ax.plot(acts[k,:,0],acts[k,:,1],acts[k,:,2],'k')

			x = covariance[p,q,0]*npy.outer(npy.cos(u),npy.sin(v)) + expected_vel[p,q,0]
			y = covariance[p,q,1]*npy.outer(npy.sin(u),npy.sin(v)) + expected_vel[p,q,1]
			z = covariance[p,q,2]*npy.outer(npy.ones_like(u),npy.cos(v)) + expected_vel[p,q,2]

			ax.plot_surface(x,y,z,color='r',alpha=0.5,linewidth=0.1)
			# ax.quiver(0,0,0,expected_vel[p,q,0],expected_vel[p,q,1],expected_vel[p,q,2],cmap='Reds',pivot='tail',linewidth=4,length=1)			
			# ax.quiver(0,0,0,action_space[q,0],action_space[q,1],action_space[q,2],cmap='Blues',pivot='tail',linewidth=4,length=1)			

			ax.quiver(0,0,0,expected_vel[p,q,0],expected_vel[p,q,1],expected_vel[p,q,2],colors='r',pivot='tail',linewidth=3,length=1)			
			ax.quiver(0,0,0,action_space[q,0],action_space[q,1],action_space[q,2],colors='b',pivot='tail',linewidth=3,length=1)			
			ax.set_xlabel("X Axis")
			ax.set_ylabel("Y Axis")
			ax.set_zlabel("Z Axis")
			plt.title("Trajectory: {0}, Action: {1}".format(p,action_space[q]))
			plt.savefig("Traj_{0}_Action_{1}_WithCov_Nolines.png".format(p,q),bbox_inches='tight')
			plt.close()
			# plt.show()

if __name__ == '__main__':
	main(sys.argv)

