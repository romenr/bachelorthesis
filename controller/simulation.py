#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool, Int32, Float32MultiArray
from sensor_msgs.msg import Image
from parameters import *


# Interface to the Simulation. This class is a ros node and handles all communication to the Simulation.
class Simulation:

	def __init__(self):
		self.bridge = CvBridge()

		# Simulation variables
		self.img = None
		self.img_set = False
		self.terminate = False
		self.target_last_seen = 0
		self.angle_to_target = 0.0
		self.path_complete = False
		self.collision = False
		self.prox_sensor_data = np.zeros(5)

		# Ros Node snake_controller setup
		# Control the Snake by publishing the Radius OR Angle Publisher
		self.radius_pub = rospy.Publisher('turningRadius', Float32, queue_size=1)
		self.angle_pub = rospy.Publisher('turningAngle', Float32, queue_size=1)
		self.select_path_pub = rospy.Publisher('selectPath', Int32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)

		self.image_sub = rospy.Subscriber('redImage', Image, self.image_callback)
		self.path_completed_sub = rospy.Subscriber('completedPath', Bool, self.path_completed_callback)
		self.angle_to_target_sub = rospy.Subscriber('angleToTarget', Float32, self.angle_to_target_callback)
		self.collision_sub = rospy.Subscriber('collision', Bool, self.collision_callback)
		self.prox_sub = rospy.Subscriber('prox', Float32MultiArray, self.prox_sensor_callback)

		rospy.init_node('snake_controller')
		self.rate = rospy.Rate(rate)

	def reset(self):
		self.target_last_seen = 0
		self.terminate = False
		self.path_complete = False
		self.collision = False
		self.angle_to_target = 0
		self.img_set = False
		self.prox_sensor_data = np.zeros(5)

		self.radius_pub.publish(0.0)
		self.reset_pub.publish(True)

	def publish_action(self, angle):
		self.angle_pub.publish(angle)
		self.rate.sleep()

	def image_callback(self, msg):
		cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
		cv_image = cv.flip(cv_image, 0)[:, :, 0]			# get red (temperature) channel
		# Put the default temperature to 0 and normalize to [0; 1].
		# Temperatures lower than default get mapped to [-1; 0]
		self.img = (cv_image - default_temperature) / (256. - default_temperature)
		self.img_set = True
		m = cv.moments(self.img, True)						# compute image moments
		if m['m00'] == 0:
			# If there is nothing visible
			self.target_last_seen += 1
			if self.target_last_seen >= reset_steps:
				self.terminate = True
		else:
			self.target_last_seen = 0

		# Show the black and white image that the snake sees
		dst = cv.resize(self.img, (200, 200))
		cv.imshow('Infrared vision', dst)
		cv.waitKey(2)

	def angle_to_target_callback(self, msg):
		self.angle_to_target = -msg.data  # Why -msg.data

	def path_completed_callback(self, msg):
		if msg.data:
			print "Path completed resetting simulation ..."
			self.terminate = True
			self.path_complete = True

	def collision_callback(self, msg):
		if msg.data:
			print "Collision! resetting simulation ..."
			self.terminate = True
			self.collision = True

	def prox_sensor_callback(self, msg):
		data = np.array(msg.data)
		# Sensor can measure in [0; prox_sensor_max_dist] and -1 means nothing sensed
		data[data < 0] = 3.
		data[data > 3] = 3.
		data = np.ones(data.size) - (data / 3.)
		self.prox_sensor_data = data

