#!/usr/bin/env python

import time
import numpy as np
from parameters import *
from simulation import Simulation


class VrepEnvironment:
	def __init__(self, path, path_mirrored):
		self.sim = Simulation()

		self.steps = 0
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.path = path
		self.path_step_count = 0
		self.path_mirrored = path_mirrored
		self.mirrored = False
		resize_x = img_resolution[1]//resolution[0]
		resize_y = (img_resolution[0] - crop_top - crop_bottom)//resolution[1]
		self.resize_factor = np.array([resize_x, resize_y])

		# Publish initial path
		self.update_path()

	def update_path(self):
		if self.mirrored:
			self.sim.select_path_pub.publish(self.path)
		else:
			self.sim.select_path_pub.publish(self.path_mirrored)

	def reset(self):
		# Reset model
		print "Terminate episode after", self.steps, "steps"
		# Change path only if there is at least half the progress on this path than on the other one
		if self.steps > self.path_step_count / 2:
			self.path_step_count = self.steps
			self.mirrored = not self.mirrored
			self.update_path()
		self.steps = 0
		self.turn_pre = 0.0
		self.sim.reset()
		time.sleep(1)
		return self.get_state_zero()

	def step(self, snn_output):

		if self.sim.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		angle = self.get_turning_angle(snn_output)
		reward = self.get_relative_reward(snn_output)
		self.sim.publish_action(angle, snn_output[velocity_neuron])

		s = self.get_state()					# New state
		a = self.sim.angle_to_target				# Angle to target (error angle)
		r = reward			# Received reward
		t = self.sim.terminate						# Episode Terminates
		n = self.steps							# Current step
		p = self.sim.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_linear_reward(self):
		velocity_reward = (self.sim.distance_to_target - d_target) / d_target
		return np.array([-self.sim.angle_to_target, self.sim.angle_to_target, velocity_reward]) * reward_factor

	def get_event_based_reward(self):
		velocity_reward = 0
		if self.sim.distance_to_target < 3.:
			velocity_reward = -1
		if self.sim.distance_to_target > 5.5:
			velocity_reward = 1
		return np.array([-self.sim.angle_to_target, self.sim.angle_to_target, velocity_reward]) * reward_factor

	def get_relative_reward(self, snn_output):
		target = n_max * self.sim.angle_to_target / a_max
		r = ((target + snn_output[0]) - snn_output[1])/n_max
		velocity_reward = (self.sim.distance_to_target - d_target) / d_target
		reward = np.array([-r, r, velocity_reward]) * reward_factor
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
		if self.sim.img_set:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):
					new_state[x//self.resize_factor[0], y//self.resize_factor[1]] += self.sim.img[y + crop_top, x]

		return [new_state / np.prod(self.resize_factor, dtype=float), self.sim.distance_to_target]

	def get_state_zero(self):
		return [np.zeros((resolution[0], resolution[1]), dtype=int), d_target], np.zeros(output_layer_size)
