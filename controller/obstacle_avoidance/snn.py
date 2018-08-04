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
		# TODO Check what exactly these do (I think they prevent generators from firing multiple times at once)
		self.input_layer = nest.Create("parrot_neuron", input_layer_size)
		nest.Connect(self.spike_generators, self.input_layer, "one_to_one")

		# HIDDEN LAYER
		self.hidden_layer = nest.Create("iaf_psc_alpha", hidden_layer_size, params=iaf_params)

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
		nest.Connect(self.input_layer, self.hidden_layer, "all_to_all", syn_spec=r_stdp_synapse_options)
		nest.Connect(self.hidden_layer, self.output_layer, "all_to_all", syn_spec=r_stdp_synapse_options)

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer[0]])
		self.conn_r = nest.GetConnections(target=[self.output_layer[1]])
		self.input_hidden_con = []
		for i in range(input_layer_size):
			for j in range(hidden_layer_size):
				con = nest.GetConnections(target=[self.hidden_layer[j]], source=[self.input_layer[i]])
				self.input_hidden_con.append(con)

	def set_reward(self, reward):
		# Set reward signal for left and right network
		nest.SetStatus(self.conn_l, {"n": -reward * reward_factor})
		nest.SetStatus(self.conn_r, {"n": reward * reward_factor})

	def simulate(self, state):
		time = nest.GetKernelStatus("time")
		nest.SetStatus(self.spike_generators, {"origin": time})
		nest.SetStatus(self.spike_generators, {"stop": sim_time_step})

		# Map state to poison spike generators
		# Every value of state needs to be in the range [0;1] to be mapped to the [min, max] firing rate
		state = state.reshape(state.size)
		poisson_rate = np.multiply(np.clip(state, 0, 1), max_poisson_freq)
		for i, r in enumerate(poisson_rate):
			nest.SetStatus([self.spike_generators[i]], {"rate": r})

		# Simulate network
		nest.Simulate(sim_time_step)
		# Get left and right output spikes
		n_l = nest.GetStatus(self.spike_detector, keys="n_events")[0]
		n_r = nest.GetStatus(self.spike_detector, keys="n_events")[1]

		# Reset output spike detector
		nest.SetStatus(self.spike_detector, {"n_events": 0})

		# Get network weights
		weights_l = np.array(nest.GetStatus(self.conn_l, keys="weight"))
		weights_r = np.array(nest.GetStatus(self.conn_r, keys="weight"))
		return n_l, n_r, weights_l, weights_r

	def set_weights(self, weights_l, weights_r):
		w_l = [{'weight': w} for w in weights_l.reshape(weights_l.size)]
		w_r = [{'weight': w} for w in weights_r.reshape(weights_r.size)]
		nest.SetStatus(self.conn_l, w_l)
		nest.SetStatus(self.conn_r, w_r)
