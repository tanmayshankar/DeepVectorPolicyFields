#!/usr/bin/env python 

import numpy as npy
import pcl
import pypcd
import sys

def base_index(point):
	lower = npy.array([-1.,-1.,0.])
	gcs = 0.02*npy.ones(3)
	return npy.floor((point-lower)/gcs)

def main(args):    

	# Should be an XYZRGB pointcloud.
	# file_list = npy.load(str(sys.argv[1]))
	
	for i in range(1):		
		for j in range(len(file_list[i])):
		

			print(i,j)
			pc = pcl.load(file_list[i][j])

			# Kill Z values beyond 1.8 m
			passthrough_filter = pc.make_passthrough_filter()
			passthrough_filter.set_filter_field_name('z')
			# passthrough_filter.set_filter_limits(0,1.8)
			passthrough_filter.set_filter_limits(0.5,1.8)
				pc = passthrough_filter.filter()

			# Remove outliers
			outlier_filter = pc.make_statistical_outlier_filter()
			outlier_filter.set_mean_k(50)
			outlier_filter.set_std_dev_mul_thresh(2)
			pc = outlier_filter.filter()

			voxel_filter = pc.make_voxel_grid_filter()
			voxel_filter.set_leaf_size(0.01,0.01,0.01)
			pc = voxel_filter.filter()

			# Save both a pointcloud and an array. 
			pcl.save(pc,"D{0}_PCD/PointCloud_{1}.pcd".format(i+2,j),binary=True)
			npy.save("D{0}_PCD/PointCloud_{1}.npy".format(i+2,j),pc.to_array())

	maxmin = npy.zeros((301,6))

	for j in range(301):
		print(j)
		pc = pcl.load("PointCloud_{0}.pcd".format(j))

		vox = pc.make_voxel_grid_filter()
		vox.set_leaf_size(0.01,0.01,0.01)
		pc = vox.filter()

		pclim = pc.to_array()

		maxmin[j,:3] = npy.max(pclim[:,:3],axis=0)
		maxmin[j,3:] = npy.min(pclim[:,:3],axis=0)

	npy.save("Maxmin.npy",maxmin)




#################################
	# for j in range(301):

	# 	print("Processing:",j)

	# 	pc = pcl.load("PointCloud_{0}.pcd".format(j))

	# 	vox = pc.make_voxel_grid_filter()
	# 	vox.set_leaf_size(0.01,0.01,0.01)
	# 	pc = vox.filter()

	# 	pcl.save(pc,"VoxelFiltered_{0}.pcd".format(j),binary=True)

	# 	pclim = pc.to_array()
	# 	pclim[:,2] -= 0.5

	# 	norm = npy.array([1.,0.6,1.3])
	# 	pclim[:,:3] /= norm

	# 	pc_array = npy.zeros((101,101,66,4))
	# 	pointrgb_array = npy.zeros((101,101,66,3))

	# 	for i in range(len(pclim)):
	# 		inds = base_index(pclim[i,:3]).astype(int)
	# 		pc_array[inds[0],inds[1],inds[2],:3] += pclim[i,3:]
	# 		pc_array[inds[0],inds[1],inds[2],3] += 1

	# 	for i in range(101):
	# 		for j in range(101):
	# 			for k in range(66):
	# 				if pc_array[i,j,k,3]:
	# 					pointrgb_array[i,j,k] = pc_array[i,j,k,:3]/pc_array[i,j,k,3]

	# 	npy.save("Voxelized_PC{0}.npy".format(j),pointrgb_array)

# ##############################
# 	for ji in range(301):

# 		print("Processing:",ji)

# 		pc = pcl.load("VoxelFiltered_{0}.pcd".format(ji))
# 		pclim = pc.to_array()
# 		pclim[:,2] -= 0.5

# 		norm = npy.array([1.,0.6,1.3])
# 		pclim[:,:3] /= norm

# 		pc_array = npy.zeros((101,101,66,4))
# 		pointrgb_array = npy.zeros((3,101,101,66))

# 		for i in range(len(pclim)):
# 			inds = base_index(pclim[i,:3]).astype(int)
# 			pc_array[inds[0],inds[1],inds[2],:3] += pclim[i,3:]
# 			pc_array[inds[0],inds[1],inds[2],3] += 1

# 		for i in range(101):
# 			for j in range(101):
# 				for k in range(66):
# 					if pc_array[i,j,k,3]:
# 						pointrgb_array[:,i,j,k] = pc_array[i,j,k,:3]/pc_array[i,j,k,3]

# 		npy.save("Voxel_TFX_PC{0}.npy".format(ji),pointrgb_array)

# #####################
##############################
	for ji in range(301):

		print("Processing:",ji)
		pc_array = npy.load("Voxelized_PC{0}.npy".format(ji))
		pointrgb_array = npy.zeros((3,101,101,66))
		for i in range(101):
			for j in range(101):
				for k in range(66):					
					pointrgb_array[:,i,j,k] = pc_array[i,j,k,:]

		npy.save("Voxel_TFX_PC{0}.npy".format(ji),pointrgb_array)

if __name__ == '__main__':
	main(sys.argv)

