#!/usr/bin/env python

import numpy as np
from parameters import *
from snn import TargetFollowingSNN, ObstacleAvoidanceSNN, nest_simulate


class Model:

	def __init__(self):
		self.snn_tf = TargetFollowingSNN()
		self.snn_oa = ObstacleAvoidanceSNN()
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.weights_tf = []
		self.weights_oa = []

	def reset(self):
		self.turn_pre = 0.0
		self.angle_pre = 0.0

	def simulate(self, state):
		self.snn_tf.set_input(state)
		self.snn_oa.set_input(state)

		# Simulate both networks
		nest_simulate()

		output, self.weights_tf = self.snn_tf.get_results()
		output_p, self.weights_oa = self.snn_oa.get_results()

		angle = self.get_turning_angle(output)
		angle_oa = self.get_obstacle_avoidance_angle(output_p)

		if np.any(state["prox"][1:] > 0.25) and not (
						abs(angle) > abs(angle_oa) and np.sign(angle) == np.sign(angle_oa)):
			angle = angle_oa
		return angle

	def get_turning_angle(self, snn_output):
		# Snake turning model
		m_l = snn_output[left_neuron]
		m_r = snn_output[right_neuron]
		angle = a_max * (m_l - m_r)
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.turn_pre = c * angle + (1 - c) * self.turn_pre
		return self.turn_pre

	def get_obstacle_avoidance_angle(self, snn_output):
		m_l = snn_output[left_neuron]
		m_r = snn_output[right_neuron]
		angle = a_avoidance_max * (m_l - m_r)
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
