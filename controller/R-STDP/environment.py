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
		self.path_complete = False
		self.steps = 0
		self.target_last_seen = 0
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.angle_to_target = 0.0
		self.path = path
		self.path_mirrored = path_mirrored
		self.mirrored = False

		# Ros Node rstdp_controller setup
		# Control the Snake by publishing the Radius OR Angle Publisher
		self.radius_pub = rospy.Publisher('turningRadius', Float32, queue_size=1)
		self.angle_pub = rospy.Publisher('turningAngle', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
		self.select_path_pub = rospy.Publisher('selectPath', Int32, queue_size=1)
		self.image_sub = rospy.Subscriber('redImage', Image, self.image_callback)
		self.path_completed_sub = rospy.Subscriber('completedPath', Bool, self.path_completed)
		self.angle_to_target_sub = rospy.Subscriber('angleToTarget', Float32, self.angle_to_target_callback)
		rospy.init_node('rstdp_controller')
		self.rate = rospy.Rate(rate)

		# Publish initial path
		self.update_path()

	def image_callback(self, msg):
		cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
		cv_image = cv.flip(cv_image, 0)[:, :, 0]			# get red (temperature) channel
		self.img = cv_image - default_temperature			# Put the default temperature to 0
		self.img_set = True
		M = cv.moments(self.img, True)						# compute image moments
		if M['m00'] == 0:
			# If there is nothing visible
			self.target_last_seen += 1
			if self.target_last_seen >= reset_steps:
				self.terminate = True
		else:
			self.target_last_seen = 0

		# Show the black and white image that the snake sees
		dst = cv.resize(self.img, (200, 200))
		cv.imshow('image', dst)
		cv.waitKey(2)

		# Interpret and show state as a black and white image
		state = self.get_state()
		state = np.swapaxes(state, 0, 1)
		state = np.interp(state, (state.min(), state.max()), (-1, +1))
		im = np.array(state * 255, dtype=np.uint8)
		img = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
		cv.imshow("state", img)
		cv.waitKey(2)

	def angle_to_target_callback(self, msg):
		self.angle_to_target = -msg.data  # Why -msg.data

	def path_completed(self, msg):
		print "Path completed resetting simulation ..."
		self.terminate = True
		self.path_complete = True
		self.mirrored = not self.mirrored

	def update_path(self):
		if self.mirrored:
			self.select_path_pub.publish(self.path)
		else:
			self.select_path_pub.publish(self.path_mirrored)

	def reset(self):
		# Reset model
		print "Terminate episode after", self.steps, "steps"
		self.steps = 0
		self.target_last_seen = 0
		self.terminate = False
		self.path_complete = False
		self.turn_pre = 0.0
		self.radius_pub.publish(0.0)
		self.reset_pub.publish(True)
		self.update_path()
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

	def step(self, n_l, n_r):

		if self.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		angle = self.get_turning_angle(n_l, n_r)
		self.angle_pub.publish(angle)
		self.rate.sleep()

		s = self.get_state()					# New state
		a = self.angle_to_target				# Angle to target (error angle)
		r = self.get_linear_reward()			# Received reward
		t = self.terminate						# Episode Terminates
		n = self.steps							# Current step
		p = self.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_linear_reward(self):
		return self.angle_to_target

	def get_turning_angle(self, n_l, n_r):
		# Snake turning model
		m_l = n_l/n_max
		m_r = n_r/n_max
		angle = a_max * (m_l - m_r)
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.turn_pre = c * angle + (1 - c) * self.turn_pre
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
		new_state = np.zeros((resolution[0], resolution[1]), dtype=int)  # 8x4
		# bring the red filtered image in the form of the state
		if self.img_set:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):				
					if self.img[y + crop_top, x] > 0:
						xpos = x//(img_resolution[1]//resolution[0])
						ypos = y//((img_resolution[0] - crop_top - crop_bottom)//resolution[1])
						new_state[xpos, ypos] += 1
		return new_state
