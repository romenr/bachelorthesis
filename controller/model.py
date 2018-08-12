#!/usr/bin/env python

from parameters import *
from snn import SpikingNeuralNetwork


class Model:

	def __init__(self):
		self.snn = SpikingNeuralNetwork()
		self.turn_pre = 0.0
		self.angle_pre = 0.0
		self.weights = []

	def reset(self):
		self.turn_pre = 0.0
		self.angle_pre = 0.0

	def simulate(self, state, reward):
		self.snn.set_reward(reward)
		output, self.weights = self.snn.simulate(state)
		angle = self.get_turning_angle(output)
		action = dict(angle=angle, velocity=output[velocity_neuron], left=output[left_neuron], right=output[right_neuron])
		return action

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
