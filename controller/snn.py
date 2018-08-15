#!/usr/bin/env python

import nest
import numpy as np
from parameters import *


class SpikingNeuralNetwork:
	def __init__(self):
		nest.set_verbosity('M_WARNING')
		nest.ResetKernel()
		nest.SetKernelStatus(nest_kernel_status)

		# INPUT LAYER
		# The spike generators are the input of the snn
		self.spike_generators = nest.Create("poisson_generator", input_layer_size, params=poisson_params)
		# The parrot neuron repeats the same poisson spike train to each connected neuron
		self.input_layer = nest.Create("parrot_neuron", input_layer_size)
		nest.Connect(self.spike_generators, self.input_layer, "one_to_one")

		# OUTPUT LAYER
		# Create motor IAF neurons
		self.output_layer = nest.Create("iaf_psc_alpha", output_layer_size, params=iaf_params)
		# Create Output spike detector
		self.spike_detector = nest.Create("spike_detector", output_layer_size, params={"withtime": True})
		nest.Connect(self.output_layer, self.spike_detector, "one_to_one")

		# Create R-STDP all to all connection
		self.vt = nest.Create("volume_transmitter")
		r_stdp_synapse_defaults = {
			"vt": self.vt[0],
			"tau_c": tau_c,
			"tau_n": tau_n,
			"Wmin": w_min,
			"Wmax": w_max,
			"A_plus": A_plus,
			"A_minus": A_minus
		}
		nest.SetDefaults("stdp_dopamine_synapse", r_stdp_synapse_defaults)
		nest.Connect(self.input_layer, self.output_layer, "all_to_all", syn_spec=r_stdp_synapse_options)

		# Print network for debugging
		# nest.PrintNetwork(depth=6)

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer[left_neuron]])
		self.conn_r = nest.GetConnections(target=[self.output_layer[right_neuron]])

	def set_reward(self, reward):
		# Set reward signal for left and right network
		nest.SetStatus(self.conn_l, {"n": reward[left_neuron]})
		nest.SetStatus(self.conn_r, {"n": reward[right_neuron]})

	def simulate(self, state):
		time = nest.GetKernelStatus("time")
		nest.SetStatus(self.spike_generators, {"origin": time})
		nest.SetStatus(self.spike_generators, {"stop": sim_time_step})

		# Map state to poison spike generators
		# Every value of state needs to be in the range [0;1] to be mapped to the [min, max] firing rate
		image = state['image']
		image = image.reshape(image.size)
		poisson_rate = np.multiply(np.clip(image, 0, 1), max_poisson_freq)
		for i, r in enumerate(poisson_rate):
			nest.SetStatus([self.spike_generators[i]], {"rate": r})

		# Simulate network
		nest.Simulate(sim_time_step)
		# Get left and right output spikes [left, right]
		output = np.array(nest.GetStatus(self.spike_detector, keys="n_events"))
		output = output / n_max

		# Reset output spike detector
		nest.SetStatus(self.spike_detector, {"n_events": 0})

		# Get network weights
		weights_l = np.array(nest.GetStatus(self.conn_l, keys="weight")).reshape(resolution)
		weights_r = np.array(nest.GetStatus(self.conn_r, keys="weight")).reshape(resolution)
		weights = [weights_l, weights_r]
		return output, weights

	def set_weights(self, weights_l, weights_r):
		w_l = [{'weight': w} for w in weights_l]
		w_r = [{'weight': w} for w in weights_r]
		nest.SetStatus(self.conn_l, w_l)
		nest.SetStatus(self.conn_r, w_r)
