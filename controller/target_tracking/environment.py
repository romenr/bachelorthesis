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
		self.distance_to_target = d_target
		self.path = path
		self.path_step_count = 0
		self.path_mirrored = path_mirrored
		self.mirrored = False
		self.resize_factor = (img_resolution[1]//resolution[0],
							(img_resolution[0] - crop_top - crop_bottom)//resolution[1])

		# Ros Node rstdp_controller setup
		# Control the Snake by publishing the Radius OR Angle Publisher
		self.radius_pub = rospy.Publisher('turningRadius', Float32, queue_size=1)
		self.angle_pub = rospy.Publisher('turningAngle', Float32, queue_size=1)
		self.velocity_pub = rospy.Publisher('velocity', Float32, queue_size=1)
		self.select_path_pub = rospy.Publisher('selectPath', Int32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
		self.image_sub = rospy.Subscriber('redImage', Image, self.image_callback)
		self.path_completed_sub = rospy.Subscriber('completedPath', Bool, self.path_completed)
		self.angle_to_target_sub = rospy.Subscriber('angleToTarget', Float32, self.angle_to_target_callback)
		self.distance_to_target_sub = rospy.Subscriber('distanceToTarget', Float32, self.distance_to_target_callback)
		rospy.init_node('rstdp_controller')
		self.rate = rospy.Rate(rate)

		# Publish initial path
		self.update_path()

	def image_callback(self, msg):
		cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
		cv_image = cv.flip(cv_image, 0)[:, :, 0]			# get red (temperature) channel
		# Put the default temperature to 0 and normalize to [0; 1].
		# Temperatures lower than default get mapped to [-1; 0]
		self.img = (cv_image - default_temperature) / (256. - default_temperature)
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
		cv.imshow('Infrared vision', dst)
		cv.waitKey(2)

		# Interpret and show state as a black and white image
		state = self.get_state()[image_index]
		state = np.swapaxes(state, 0, 1)
		cv.imshow("State", np.array(state))
		cv.waitKey(2)

	def angle_to_target_callback(self, msg):
		self.angle_to_target = -msg.data  # Why -msg.data

	def distance_to_target_callback(self, msg):
		self.distance_to_target = msg.data

	def path_completed(self, msg):
		print "Path completed resetting simulation ..."
		self.terminate = True
		self.path_complete = True

	def update_path(self):
		if self.mirrored:
			self.select_path_pub.publish(self.path)
		else:
			self.select_path_pub.publish(self.path_mirrored)

	def reset(self):
		# Reset model
		print "Terminate episode after", self.steps, "steps"
		# Change path only if there is at least half the progress on this path than on the other one
		if self.steps > self.path_step_count / 2:
			self.path_step_count = self.steps
			self.mirrored = not self.mirrored
			self.update_path()
		self.steps = 0
		self.target_last_seen = 0
		self.terminate = False
		self.path_complete = False
		self.turn_pre = 0.0
		self.distance_to_target = d_target
		self.angle_to_target = 0
		self.radius_pub.publish(0.0)
		self.velocity_pub.publish(0.5)
		self.reset_pub.publish(True)
		time.sleep(1)
		return [np.zeros((resolution[0], resolution[1]), dtype=int), d_target], np.zeros(output_layer_size)

	def step(self, snn_output):

		if self.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		angle = self.get_turning_angle(snn_output)
		self.angle_pub.publish(angle)

		self.velocity_pub.publish(snn_output[velocity_neuron])
		self.rate.sleep()

		s = self.get_state()					# New state
		a = self.angle_to_target				# Angle to target (error angle)
		r = self.get_linear_reward()			# Received reward
		t = self.terminate						# Episode Terminates
		n = self.steps							# Current step
		p = self.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_linear_reward(self):
		velocity_reward = (self.distance_to_target - d_target) / d_target
		return np.array([-self.angle_to_target, self.angle_to_target, velocity_reward]) * reward_factor

	def get_relative_reward(self, snn_output):
		target = n_max * self.angle_to_target / a_max
		r = ((target + snn_output[0]) - snn_output[1])/n_max
		reward = np.array([-r, r]) * reward_factor
		return reward

	def get_turning_angle(self, snn_output):
		# Snake turning model
		m_l = snn_output[left_neuron]
		m_r = snn_output[right_neuron]
		angle = a_max * (m_l - m_r)
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.turn_pre = c * angle + (1 - c) * self.turn_pre
		return angle

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
		return [new_state / float(np.prod(self.resize_factor)), self.distance_to_target]
