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


def connect_all_to_all_r_stdp(first_layer, second_layer):
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


class TargetFollowingSNN:
	def __init__(self):
		self.spike_generators, self.input_layer = create_input_layer(input_layer_size)
		self.output_layer, self.spike_detectors = create_output_layer(output_layer_size)
		connect_all_to_all_r_stdp(self.input_layer, self.output_layer)

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
		output = np.array(nest.GetStatus(self.spike_detectors, keys="n_events"))
		output = output / n_max

		# Reset output spike detector
		nest.SetStatus(self.spike_detectors, {"n_events": 0})

		# Get network weights
		weights_l = np.array(nest.GetStatus(self.conn_l, keys="weight")).reshape(resolution)
		weights_r = np.array(nest.GetStatus(self.conn_r, keys="weight")).reshape(resolution)
		weights = [weights_l, weights_r]
		return output, weights

	def set_weights(self, weights_l, weights_r):
		w_l = [{'weight': w} for w in weights_l.reshape(weights_l.size)]
		w_r = [{'weight': w} for w in weights_r.reshape(weights_r.size)]
		nest.SetStatus(self.conn_l, w_l)
		nest.SetStatus(self.conn_r, w_r)


class ObstacleAvoidanceSNN:
	def __init__(self):
		self.spike_generators, self.input_layer = create_input_layer(4)
		self.output_layer, self.spike_detectors = create_output_layer(2)
		# Connect the right proximity sensors to the left neuron and the other way around
		connect_all_to_all_r_stdp(self.input_layer[2:], [self.output_layer[left_neuron]])
		connect_all_to_all_r_stdp(self.input_layer[:2], [self.output_layer[right_neuron]])

		# Create connection handles
		self.conn_l = nest.GetConnections(target=[self.output_layer[left_neuron]])
		self.conn_r = nest.GetConnections(target=[self.output_layer[right_neuron]])

	def set_reward(self, reward):
		# Set reward signal for left and right network
		nest.SetStatus(self.conn_l, {"n": reward[2]})
		nest.SetStatus(self.conn_r, {"n": reward[3]})

	def simulate(self, state):
		time = nest.GetKernelStatus("time")
		nest.SetStatus(self.spike_generators, {"origin": time})
		nest.SetStatus(self.spike_generators, {"stop": sim_time_step})

		# Map state to poison spike generators
		# Every value of state needs to be in the range [0;1] to be mapped to the [min, max] firing rate
		prox_data = state['prox']
		poisson_rate = np.multiply(np.clip(prox_data[1:], 0, 1), max_poisson_freq)
		for i, r in enumerate(poisson_rate):
			nest.SetStatus([self.spike_generators[i]], {"rate": r})

		# Simulate network
		nest.Simulate(sim_time_step)
		# Get left and right output spikes [left, right]
		output = np.array(nest.GetStatus(self.spike_detectors, keys="n_events"))
		output = output / n_max

		# Reset output spike detector
		nest.SetStatus(self.spike_detectors, {"n_events": 0})

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

