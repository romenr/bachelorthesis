#!/usr/bin/env python

import numpy as np
from parameters import *
from snn import SpikingNeuralNetwork
from network import ProxSpikingNeuralNetwork


class Model:

	def __init__(self):
		self.snn = SpikingNeuralNetwork()
		self.psnn = ProxSpikingNeuralNetwork()
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.weights = []
		self.weigts_p = []

	def reset(self):
		self.turn_pre = 0.0
		self.angle_pre = 0.0

	def simulate(self, state, reward):
		if reward is not None:
			# self.snn.set_reward(reward)
			self.psnn.set_reward(reward)
		output, self.weights = self.snn.simulate(state)
		output_p, self.weigts_p = self.psnn.simulate(state)
		print output_p
		angle = self.get_turning_angle(output)

		if np.any(state["prox"][1:] > 0.25):
			angle = self.get_obstacle_avoidance_angle(output_p)
		action = dict(angle=angle, left=output[left_neuron], right=output[right_neuron])
		return action

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
