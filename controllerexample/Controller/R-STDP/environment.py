#!/usr/bin/env python

import sys
import rospy
import math
import time
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Image
from parameters import *

sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert


class VrepEnvironment:
	def __init__(self):
		self.image_sub = rospy.Subscriber('redImage', Image, self.image_callback)
		# You can controll the Snake by publishing the Radius OR Angle Publisher
		self.radius_pub = rospy.Publisher('turningRadius', Float32, queue_size=1)
		self.angle_pub = rospy.Publisher('turningAngle', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
		self.img = None
		self.imgFlag = False
		self.cx = 0.0
		self.terminate = False
		self.startLeft = True
		self.steps = 0
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.bridge = CvBridge()
		rospy.init_node('rstdp_controller')
		self.rate = rospy.Rate(rate)

	def image_callback(self, msg):
		# Process incoming image data

		# Get an OpenCV image
		cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
		self.img = cv_image[:, :, 2]					# get red channel
		M = cv.moments(self.img, True)			# compute image moments for centroid
		if M['m00'] == 0:
			self.terminate = True
			self.cx = 0.0
		else:
			self.terminate = False
			# normalized centroid position
			self.cx = 2*M['m10']/(M['m00']*img_resolution[1]) - 1.0

		dst = cv.resize(self.img, (200, 200))
		cv.imshow('image', dst)
		cv.waitKey(2)

		self.imgFlag = True

		return

	def reset(self):
		print "Reset after ", self.steps, " Steps"
		# Reset model
		self.turn_pre = 0.0
		self.radius_pub.publish(0.0)
		# Change lane
		self.startLeft = not self.startLeft
		self.reset_pub.publish(Bool(self.startLeft))
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

	def step(self, n_l, n_r):

		self.steps += 1

		# Publish turning radius
		# radius = self.get_turning_radius(n_l, n_r)
		# self.radius_pub.publish(radius)

		# Publish turning angle
		angle = self.get_turning_angle(n_l, n_r)
		self.angle_pub.publish(angle)
		self.rate.sleep()

		# Set reward signal
		r = self.get_linear_reward()

		s = self.get_state()
		n = self.steps
		lane = self.startLeft

		# Terminate episode of robot reaches start position again
		# or reset distance
		t = self.terminate
		if t:
			self.reset()
			self.steps = 0
			self.terminate = False

		# Return state, distance, position, reward, termination, steps, lane
		return s, self.cx, r, t, n, lane

	def get_turning_angle(self, n_l, n_r):
		# Snake turning model
		m_l = n_l/n_max
		m_r = n_r/n_max
		a_l = m_l * a_max
		a_r = m_r * -a_max
		angle = a_l + a_r
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.angle_pre = c * angle + (1 - c) * self.angle_pre
		# print c, angle, self.angle_pre
		return self.angle_pre

	def get_turning_radius(self, n_l, n_r):
		# Snake turning model
		m_l = n_l/n_max
		m_r = n_r/n_max
		a = m_l - m_r
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.turn_pre = c*0.5*a + (1-c)*self.turn_pre
		if abs(self.turn_pre) < 0.001:
			radius = 0
		else:
			radius = r_min/self.turn_pre
		return radius

	def get_state(self):
		new_state = np.zeros((resolution[0], resolution[1]), dtype=int)  # 8x4
		# bring the red filtered image in the form of the state
		if self.imgFlag:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):				
					if self.img[y + crop_top, x] > 0:
						xpos = x//(img_resolution[1]//resolution[0])
						ypos = y//((img_resolution[0] - crop_top - crop_bottom)//resolution[1])
						new_state[xpos, ypos] += 4
		return new_state

	def get_linear_reward(self):
		return self.cx
