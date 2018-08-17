#!/usr/bin/env python

import nest
import numpy as np
from parameters import *


class ProxSpikingNeuralNetwork:
	def __init__(self):

		# INPUT LAYER
		# The spike generators are the input of the snn
		self.spike_generators_l = nest.Create("poisson_generator", 2, params=poisson_params)
		# The parrot neuron repeats the same poisson spike train to each connected neuron
		self.input_layer_l = nest.Create("parrot_neuron", 2)
		nest.Connect(self.spike_generators_l, self.input_layer_l, "one_to_one")

		# The spike generators are the input of the snn
		self.spike_generators_r = nest.Create("poisson_generator", 2, params=poisson_params)
		# The parrot neuron repeats the same poisson spike train to each connected neuron
		self.input_layer_r = nest.Create("parrot_neuron", 2)
		nest.Connect(self.spike_generators_r, self.input_layer_r, "one_to_one")

		# OUTPUT LAYER
		# Create motor IAF neurons
		self.output_layer_l = nest.Create("iaf_psc_alpha", 1, params=iaf_params)
		# Create Output spike detector
		self.spike_detector_l = nest.Create("spike_detector", 1, params={"withtime": True})
		nest.Connect(self.output_layer_l, self.spike_detector_l, "one_to_one")

		self.output_layer_r = nest.Create("iaf_psc_alpha", 1, params=iaf_params)
		# Create Output spike detector
		self.spike_detector_r = nest.Create("spike_detector", 1, params={"withtime": True})
		nest.Connect(self.output_layer_r, self.spike_detector_r, "one_to_one")

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
		nest.Connect(self.input_layer_r, self.output_layer_l, "all_to_all", syn_spec=r_stdp_synapse_options)
		nest.Connect(self.input_layer_l, self.output_layer_r, "all_to_all", syn_spec=r_stdp_synapse_options)

		# Print network for debugging
		# nest.PrintNetwork(depth=6)

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer_l[0]])
		self.conn_r = nest.GetConnections(target=[self.output_layer_r[0]])

	def set_reward(self, reward):
		# Set reward signal for left and right network
		nest.SetStatus(self.conn_l, {"n": reward[2]})
		nest.SetStatus(self.conn_r, {"n": reward[3]})

	def simulate(self, state):
		time = nest.GetKernelStatus("time")
		nest.SetStatus(self.spike_generators_l, {"origin": time})
		nest.SetStatus(self.spike_generators_l, {"stop": sim_time_step})
		nest.SetStatus(self.spike_generators_r, {"origin": time})
		nest.SetStatus(self.spike_generators_r, {"stop": sim_time_step})

		# Map state to poison spike generators
		# Every value of state needs to be in the range [0;1] to be mapped to the [min, max] firing rate
		prox_data = state['prox']
		poisson_rate = np.multiply(np.clip(prox_data[1:], 0, 1), max_poisson_freq)
		for i, r in enumerate(poisson_rate[0:2]):
			nest.SetStatus([self.spike_generators_l[i]], {"rate": r})
		for i, r in enumerate(poisson_rate[2:]):
			nest.SetStatus([self.spike_generators_r[i]], {"rate": r})

		# Simulate network
		nest.Simulate(sim_time_step)
		# Get left and right output spikes [left, right]
		output_l = np.array(nest.GetStatus(self.spike_detector_l, keys="n_events"))
		output_l = output_l / n_max
		output_r = np.array(nest.GetStatus(self.spike_detector_r, keys="n_events"))
		output_r = output_r / n_max
		output = np.array([output_l[0], output_r[0]])

		# Reset output spike detector
		nest.SetStatus(self.spike_detector_l, {"n_events": 0})
		nest.SetStatus(self.spike_detector_r, {"n_events": 0})

		# Get network weights
		weights_l = np.array(nest.GetStatus(self.conn_l, keys="weight"))
		weights_r = np.array(nest.GetStatus(self.conn_r, keys="weight"))
		weights = [weights_l, weights_r]
		return output, weights

	def set_weights(self, weights_l, weights_r):
		w_l = [{'weight': w} for w in weights_l.reshape(weights_l.size)]
		w_r = [{'weight': w} for w in weights_r.reshape(weights_r.size)]
		nest.SetStatus(self.conn_l, w_l)
		nest.SetStatus(self.conn_r, w_r)
