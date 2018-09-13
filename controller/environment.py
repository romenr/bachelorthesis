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
		self.rewards_tf = []

		# Publish initial path
		self.update_path()

	def update_path(self):
		if self.mirrored:
			self.sim.collision_side_left = True
			self.sim.select_path_pub.publish(self.path)
		else:
			self.sim.collision_side_left = False
			self.sim.select_path_pub.publish(self.path_mirrored)

	def reset(self):
		# Reset model
		print "Terminate episode after", self.steps, "steps"
		# Change path only if there is at least half the progress on this path than on the other one
		self.mirrored = not self.mirrored
		self.update_path()
		self.steps = 0
		self.rewards_tf = []
		self.sim.reset()
		time.sleep(1)
		return get_state_reward_zero()

	def step(self, angle):

		if self.sim.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		self.sim.publish_action(angle)

		s = self.get_state()						# New state
		a = self.sim.angle_to_target				# Angle to target (error angle)
		r = self.get_reward()					# Received reward
		t = self.sim.terminate						# Episode Terminates
		n = self.steps								# Current step
		p = self.sim.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_reward(self):
		att = self.sim.angle_to_target * reward_factor_tf
		self.rewards_tf.append(att)
		att = np.average(self.rewards_tf[-average_window:])
		prox_reward_left, prox_reward_right = self.get_prox_reward()
		return np.array([att, -att, prox_reward_left * reward_factor_oa, prox_reward_right * reward_factor_oa])

	def get_prox_reward(self):
		if not self.sim.terminate or self.sim.path_complete:
			return 0., 0.
		if self.sim.collision_side_left:
			if self.sim.collision:
				return -1., 1.
			else:
				return 1., -1.
		else:
			if self.sim.collision:
				return 1., -1.
			else:
				return -1., 1.

	def get_state(self):
		new_state = np.zeros((resolution[0], resolution[1]), dtype=float)
		# bring the red filtered image in the form of the state
		if self.sim.img_set:
			for y in range(img_resolution[0] - crop_top - crop_bottom):
				for x in range(img_resolution[1]):
					new_state[x//self.resize_factor[0], y//self.resize_factor[1]] += self.sim.img[y + crop_top, x]

		return dict(image=new_state / np.prod(self.resize_factor, dtype=float), prox=self.sim.prox_sensor_data)


def get_state_reward_zero():
	return dict(image=np.zeros((resolution[0], resolution[1]), dtype=int), prox=np.zeros(5)), np.zeros(
		output_layer_size + 2)
