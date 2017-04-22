#!/usr/bin/env python 

import numpy as npy
import pcl
import pypcd
import sys

def main(args):    

	# Should be an XYZRGB pointcloud.
	file_list = npy.load(str(sys.argv[1]))
	
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

			# Save both a pointcloud and an array. 
			pcl.save(pc,"D{0}_PCD/PointCloud_{1}.pcd".format(i+2,j),binary=True)
			npy.save("D{0}_PCD/PointCloud_{1}.npy".format(i+2,j),pc.to_array())

if __name__ == '__main__':
	main(sys.argv)

