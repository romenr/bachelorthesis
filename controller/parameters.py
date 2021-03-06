#!/usr/bin/env python

import math

# Save file Options
default_dir = './data/default'				# Default dir if scrips are called without dir
weights_file = 'weights.h5'				# Trained weights
training_file_tf = 'tf_training_data.h5'			# Results from training
training_file_oa = 'oa_training_data.h5'			# Results from training
evaluation_file_tf = 'tf_evaluation_data.h5'		# Results from evaluation
evaluation_file_oa = 'oa_evaluation_data.h5'		# Results from evaluation

# Input image
img_resolution = [32, 32]			# Original DVS frame resolution
crop_top = 10						# Crop at the top
crop_bottom = 14					# Crop at the bottom
resolution = [16, 4]					# Resolution of reduced image

# State
image_index = 0
distance_index = 1

# Network parameters
input_layer_size = resolution[0] * resolution[1]
output_layer_size = 2				# Left, Right and Velocity neuron
left_neuron = 0
right_neuron = 1


sim_time_step = 50.0				# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.01				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max
nest_kernel_status = {				# Nest Kernel initialization options
	"local_num_threads": 1,			# Number of Threads used by nest
	"resolution": time_resolution
}

# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min_tf = 1500.					# Minimum initial random value
w0_max_tf = 1501.					# Maximum initial random value
w0_min_oa = 1.						# Minimum initial random value
w0_max_oa = 3000.					# Maximum initial random value
# These tau_n and tau_c parameters are suggested by Izhikevich, E.M. (2007). Solving the distal reward problem
# through linkage of STDP and dopamine signaling. Cereb. Cortex, 17(10), 2443-2452.
tau_n = 200.						# Time constant of reward signal
tau_c = 1000.						# Time constant of eligibility trace

reward_factor_oa = 0.01				# Reward factor modulating reward signal strength for obstacle avoidance
reward_factor_tf = 0.002			# Target following
average_window = 20					# Size of the window used for averaging
A_plus = 1.							# Constant scaling strength of potentiaion
A_minus = 1.						# Constant scaling strength of depression

r_stdp_synapse_options_tf = {					# Initialisation Options for R-STDP Synapses
	"model": "stdp_dopamine_synapse",		# R-STDP Model
	"weight": {
		"distribution": "uniform",				# Initial weight distribution
		"low": w0_min_tf,
		"high": w0_max_tf
	}
}

r_stdp_synapse_options_oa = {					# Initialisation Options for R-STDP Synapses
	"model": "stdp_dopamine_synapse",		# R-STDP Model
	"weight": {
		"distribution": "uniform",				# Initial weight distribution
		"low": w0_min_oa,
		"high": w0_max_oa
	}
}


# Snake turning model
n_max = float(sim_time_step//t_refrac)         # Maximum input activity

r_min = 3.0							# Minimum turning radius
a_max = math.pi / 2.				# Maximum turning angle
a_avoidance_max = math.pi / 4.		# Maximum obstacle avoidance angle
prox_crit_dist = 0.9				# Critical distance for obstacle avoidance
prox_sensor_max_dist = 10.				# Maximum proximity sensor sensing range

# Thermal Vision
default_temperature = 128			# Default temperature of the simulation

# Other
reset_steps = 20						# After how many steps without seeing the target should the simulation reset
rate = 20.							# ROS publication rate (step = 1/rate = 50ms)
training_length = 40000		    	# Length of training procedure (1 step ~ 50 ms)
evaluation_length = 20000			# Length of evaluation procedure

# Path numbers
plus_path = 1						# Simple path in + shape
plus_path_mirrored = 2
evaluation_path = 3					# Simple path in + shape
evaluation_path_mirrored = 4
