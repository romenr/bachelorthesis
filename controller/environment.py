#!/usr/bin/env python

import time
import numpy as np
from parameters import *
from simulation import Simulation


class VrepEnvironment:
	def __init__(self, path, path_mirrored):
		self.sim = Simulation()

		self.steps = 0
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
		self.sim.reset()
		time.sleep(1)
		return get_state_reward_zero()

	def step(self, action):

		if self.sim.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		angle = action['angle']
		reward = self.get_relative_reward(angle)
		self.sim.publish_action(angle)

		s = self.get_state(action)					# New state
		a = self.sim.angle_to_target				# Angle to target (error angle)
		r = reward			# Received reward
		t = self.sim.terminate						# Episode Terminates
		n = self.steps							# Current step
		p = self.sim.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_linear_reward(self):
		return np.array([-self.sim.angle_to_target, self.sim.angle_to_target]) * reward_factor

	def get_relative_reward(self, angle):
		r = (self.sim.angle_to_target - angle) / a_max
		reward = np.array([-r, r]) * reward_factor
		return reward

	def get_state(self, last_action):
		new_state = np.zeros((resolution[0], resolution[1]), dtype=float)
		# bring the red filtered image in the form of the state
		if self.sim.img_set:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):
					new_state[x//self.resize_factor[0], y//self.resize_factor[1]] += self.sim.img[y + crop_top, x]

		return dict(image=new_state / np.prod(self.resize_factor, dtype=float), last_action=last_action)


def get_state_reward_zero():
	return dict(image=np.zeros((resolution[0], resolution[1]), dtype=int), left=0, right=0), np.zeros(
		output_layer_size)
