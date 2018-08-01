#!/usr/bin/env python

import sys
import rospy
import time
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool, Int32
from sensor_msgs.msg import Image
from parameters import *

sys.path.append('/usr/lib/python2.7/dist-packages')


class VrepEnvironment:
	def __init__(self, path, path_mirrored):
		self.img = None
		self.img_set = False
		self.bridge = CvBridge()
		self.terminate = False
		self.distance = 0
		self.steps = 0
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.reward = 0
		self.resize_factor = (img_resolution[1]//resolution[0],
							(img_resolution[0] - crop_top - crop_bottom)//resolution[1])

		# Ros Node rstdp_controller setup
		# Control the Snake by publishing the Radius OR Angle Publisher
		self.radius_pub = rospy.Publisher('turningRadius', Float32, queue_size=1)
		self.angle_pub = rospy.Publisher('turningAngle', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
		self.image_sub = rospy.Subscriber('redImage', Image, self.image_callback)
		self.dist_sub = rospy.Subscriber('distance', Float32, self.distance_callback)
		self.reset_sub = rospy.Subscriber('reset', Bool, self.reset_callback)
		rospy.init_node('rstdp_controller')
		self.rate = rospy.Rate(rate)

	def image_callback(self, msg):
		cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
		cv_image = cv.flip(cv_image, 0)[:, :, 0]			# get red (temperature) channel
		# Put the default temperature to 0 and normalize to [0; 1].
		# Temperatures lower than default get mapped to [-1; 0]
		self.img = (cv_image - default_temperature) / (256. - default_temperature)
		self.img_set = True
		M = cv.moments(self.img, True)  # compute image moments
		if M['m00'] == 0:
			# Nothing to see nothing to learn
			self.reward = 0
		else:
			# normalized centroid position
			self.reward = 2*M['m10']/(M['m00']*img_resolution[1])
			if self.reward >= 1:
				self.reward = self.reward - 2

		# Show the black and white image that the snake sees
		dst = cv.resize(self.img, (200, 200))
		cv.imshow('Infrared vision', dst)
		cv.waitKey(2)

		# Interpret and show state as a black and white image
		state = self.get_state()
		state = np.swapaxes(state, 0, 1)
		cv.imshow("State", np.array(state))
		cv.waitKey(2)

	def distance_callback(self, msg):
		self.distance = msg.data

	def reset_callback(self, msg):
		self.terminate = True

	def reset(self):
		# Reset model
		print "Terminate episode after", self.steps, "steps"
		self.steps = 0
		self.terminate = False
		self.turn_pre = 0.0
		self.radius_pub.publish(0.0)
		self.reset_pub.publish(True)
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

	def step(self, n_l, n_r):

		if self.terminate:
			self.reset()

		self.steps += 1
		if self.steps >= episode_steps:
			self.terminate = True

		# Publish turning angle and sleep for ~50ms
		angle = self.get_turning_angle(n_l, n_r)
		self.angle_pub.publish(angle)
		self.rate.sleep()

		s = self.get_state()					# New state
		a = 0
		r = self.get_linear_reward()			# Received reward
		t = self.terminate						# Episode Terminates
		n = self.steps							# Current step
		p = False

		return s, a, r, t, n, p

	def get_linear_reward(self):
		return self.reward

	def get_turning_angle(self, n_l, n_r):
		# Snake turning model
		m_l = n_l/n_max
		m_r = n_r/n_max
		angle = a_max * (m_l - m_r)
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		# For obstacle avoidance we shouldn't continue the evasion movement if we see nothing
		self.turn_pre = (1 - c) * angle + c * self.turn_pre
		return self.turn_pre

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
		new_state = np.zeros((resolution[0], resolution[1]), dtype=float)
		# bring the red filtered image in the form of the state
		if self.img_set:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):
					new_state[x//self.resize_factor[0], y//self.resize_factor[1]] += self.img[y + crop_top, x]
		return new_state / float(np.prod(self.resize_factor))
