#!/usr/bin/env python
import roslib
import sys
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TwistStamped
import message_filters
import os

class time_sync:

	def __init__(self):

		self.FILE_DIR = FILE_DIR

		self.desired_pos = message_filters.Subscriber("/des_pos",PoseStamped)
		self.actual_pos = message_filters.Subscriber("/pos",PoseStamped)
		self.desired_vel = message_filters.Subscriber("/des_vel",TwistStamped)
		self.actual_vel = message_filters.Subscriber("/vel",TwistStamped)

		self.time_sync = message_filters.TimeSynchronizer([self.desired_pos, self.actual_pos, self.desired_vel, self.actual_vel],10)
		self.time_sync.registerCallback(self.callback)
		self.count = 0

		self.des_pos = []
		self.pos = []
		self.des_vel = []
		self.vel = []

	def callback(self, des_pos, pos, des_vel, vel):

		self.des_pos.append(des_pos)
		self.pos.append(pos)
		self.des_vel.append(vel)
		self.vel.append(vel)		
		self.count += 1

		npy.save("Desired_Pos.npy",self.des_pos)
		npy.save("Actual_Pos.npy",self.pos)
		npy.save("Desired_Vel.npy",self.des_vel)
		npy.save("Actual_Vel.npy",self.vel)

def main(argv):

	ts = time_sync()
	rospy.init_node('Tsync',anonymous=True)

	try:					
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting Down.")

if __name__ == '__main__':
	main(sys.argv)