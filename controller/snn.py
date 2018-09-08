#!/usr/bin/env python

import nest
import numpy as np
from parameters import *

print "Reset NEST kernel"
nest.set_verbosity('M_WARNING')
nest.ResetKernel()
nest.SetKernelStatus(nest_kernel_status)


def create_input_layer(n):
	"""Create a input layer with n input neurons.
	The spike generators  translate a input into spike trains.
	The parrot neuron repeats the same poisson spike train to each connected neuron.
	Otherwise each neuron gets a different input spike train from the generator.
	Set the input for the network at the poisson generators.
	Connect the next layer to the parrot neurons.
	:param n: Size of the input layer
	:return:poisson_generators, input_layer
	"""
	poisson_generators = nest.Create("poisson_generator", n, params=poisson_params)
	input_layer = nest.Create("parrot_neuron", n)
	nest.Connect(poisson_generators, input_layer, "one_to_one")
	return poisson_generators, input_layer


def create_output_layer(n):
	"""Create a output layer with n output neurons.
	Connect the output layer with previous layers.
	The output of the network can be read from the spike detectors.
	:param n: Size of the output layer
	:return: output_layer, spike_detectors
	"""
	output_layer = nest.Create("iaf_psc_alpha", n, params=iaf_params)
	# Create Output spike detector
	spike_detectors = nest.Create("spike_detector", n, params={"withtime": True})
	nest.Connect(output_layer, spike_detectors, "one_to_one")
	return output_layer, spike_detectors


def connect_all_to_all_r_stdp(first_layer, second_layer, r_stdp_synapse_options):
	"""Connect the first layer to the second layer with stdp dopamine synapses (r-stdp).
	The layers are connected all to all method.
	:param first_layer: The neurons of the first layer
	:param second_layer: The neurons of the second layer
	"""
	vt = nest.Create("volume_transmitter")
	r_stdp_synapse_defaults = {
		"vt": vt[0],
		"tau_c": tau_c,
		"tau_n": tau_n,
		"Wmin": w_min,
		"Wmax": w_max,
		"A_plus": A_plus,
		"A_minus": A_minus
	}
	nest.SetDefaults("stdp_dopamine_synapse", r_stdp_synapse_defaults)
	nest.Connect(first_layer, second_layer, "all_to_all", syn_spec=r_stdp_synapse_options)


def set_inputs(spike_generators, inputs):
	"""Reset the spike generators and set the inputs.
	The i-th input is assigned to the i-th spike_generator.
	All input values must be in [0;1]
	:param spike_generators: The spike generators
	:param inputs: Inputs for the network
	"""
	time = nest.GetKernelStatus("time")
	nest.SetStatus(spike_generators, {"origin": time})
	nest.SetStatus(spike_generators, {"stop": sim_time_step})

	poisson_rates = np.multiply(np.clip(inputs, 0, 1), max_poisson_freq)
	for i, r in enumerate(poisson_rates):
		nest.SetStatus([spike_generators[i]], {"rate": r})


def set_reward(connections, reward):
	"""Set the dopamine level in connections to reward
	:param connections: The connections to be rewarded
	:param reward: The reward
	"""
	nest.SetStatus(connections, {"n": reward})


def get_output(spike_detectors):
	"""Read the number of spikes from the spike detector and normalize them.
	Reset the spike detectors.
	:param spike_detectors: The spike detectors
	:return: Normalized firing rates of the output neurons
	"""
	output = np.array(nest.GetStatus(spike_detectors, keys="n_events"))
	nest.SetStatus(spike_detectors, {"n_events": 0})
	output = output / n_max
	return output


def set_weights(connections, weights):
	"""Set the weights on the connections.
	Size of connections and weights must be equal.
	:param connections: Connections
	:param weights: Weights
	"""
	w = [{'weight': w} for w in weights.reshape(weights.size)]
	nest.SetStatus(connections, w)


def get_weights(connections):
	"""Returns the weights of the connections
	:param connections:
	:return: Numpy array of weights
	"""
	return np.array(nest.GetStatus(connections, keys="weight"))


def nest_simulate():
	"""Simulate all networks
	"""
	nest.Simulate(sim_time_step)


class TargetFollowingSNN:
	def __init__(self):
		self.spike_generators, self.input_layer = create_input_layer(input_layer_size)
		self.output_layer, self.spike_detectors = create_output_layer(output_layer_size)
		connect_all_to_all_r_stdp(self.input_layer, self.output_layer, r_stdp_synapse_options_tf)

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer[left_neuron]])
		self.conn_r = nest.GetConnections(target=[self.output_layer[right_neuron]])

	def set_reward(self, reward):
		set_reward(self.conn_l, reward[left_neuron])
		set_reward(self.conn_r, reward[right_neuron])

	def set_input(self, state):
		image = state['image']
		image = image.reshape(image.size)
		set_inputs(self.spike_generators, image)

	def get_results(self):
		output = get_output(self.spike_detectors)
		weights_l = get_weights(self.conn_l).reshape(resolution)
		weights_r = get_weights(self.conn_r).reshape(resolution)
		return output, [weights_l, weights_r]

	def set_weights(self, weights_l, weights_r):
		set_weights(self.conn_l, weights_l)
		set_weights(self.conn_r, weights_r)


class ObstacleAvoidanceSNN:
	def __init__(self):
		self.spike_generators, self.input_layer = create_input_layer(4)
		self.output_layer, self.spike_detectors = create_output_layer(2)
		# Connect the right proximity sensors to the left neuron and the other way around
		connect_all_to_all_r_stdp(self.input_layer, self.output_layer, r_stdp_synapse_options_oa)

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer[left_neuron]])
		self.conn_r = nest.GetConnections(target=[self.output_layer[right_neuron]])

	def set_reward(self, reward):
		set_reward(self.conn_l, reward[left_neuron])
		set_reward(self.conn_r, reward[right_neuron])

	def set_input(self, state):
		set_inputs(self.spike_generators, state['prox'][1:])

	def get_results(self):
		output = get_output(self.spike_detectors)
		weights_l = get_weights(self.conn_l)
		weights_r = get_weights(self.conn_r)
		return output, [weights_l, weights_r]

	def set_weights(self, weights_l, weights_r):
		set_weights(self.conn_l, weights_l)
		set_weights(self.conn_r, weights_r)

