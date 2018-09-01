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

	def step(self, angle):

		if self.sim.terminate:
			self.reset()

		self.steps += 1

		# Publish turning angle and sleep for ~50ms
		self.sim.publish_action(angle)

		s = self.get_state()						# New state
		a = self.sim.angle_to_target				# Angle to target (error angle)
		r = self.get_linear_reward()				# Received reward
		t = self.sim.terminate						# Episode Terminates
		n = self.steps								# Current step
		p = self.sim.path_complete					# Terminated because Path was completed successfully

		return s, a, r, t, n, p

	def get_linear_reward(self):
		att = self.sim.angle_to_target * reward_factor_tf
		prox_reward_left = self.get_prox_reward(np.any(self.sim.prox_sensor_data[3:] > 0.))
		prox_reward_right = self.get_prox_reward(np.any(self.sim.prox_sensor_data[1:3] > 0.))
		return np.array([att, -att, prox_reward_left * reward_factor_oa, prox_reward_right * reward_factor_oa])

	def get_prox_reward(self, obstacle_detected):
		if not self.sim.terminate or not obstacle_detected:
			return 0
		if self.sim.collision:
			return 1
		else:
			return -1

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
