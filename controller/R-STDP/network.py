#!/usr/bin/env python

import parameters as param

nest_verbosity = 'M_WARNING'
kernel_status = {"local_num_threads": 1, "resolution": param.time_resolution}
input_size = param.view_resolution[0] * param.view_resolution[1]
output_size = 2
detector_options = {"withtime": True}
connection_options = {"model": "stdp_dopamine_synapse", "weight": {"distribution": "uniform", "low": param.init_min_weight, "high": param.init_max_weight}}

class SpikingNeuralNetwork():

	def __init__(self):
		nest.set_verbosity(nest_verbosity)
		nest.ResetKernel()
		nest.SetKernelStatus(kernel_status)
		
		self.spike_generators = nest.Create("poisson_generator", input_size)
		self.input_neurons = nest.Create("parrot_neuron", input_size)
		# Integrate and Fire neurons
		self.output_neurons = nest.Create("iaf_psc_alpha", output_size)
		self.spike_detector = nest.Create("spike_detector", output_size, params=detector_options)
		self.volume_transmitter = nest.Create("volume_transmitter")
		nest.SetDefaults("stdp_dopamine_synapse", { "vt": self.volume_transmitter[0], "tau_c": param.tau_c, "tau_n": param.tau_n, "Wmin": param.p_min, "Wmax": param.p_max, "A_plus": param.A_plus, "A_minus": param.A_minus})


