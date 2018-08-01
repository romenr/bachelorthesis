#!/usr/bin/env python

import math

# Save file Options
default_dir = './data/default'				# Default dir if scrips are called without dir
weights_file = 'weights.h5'				# Trained weights
training_file = 'training_data.h5'			# Results from training
evaluation_file = 'evaluation_data.h5'		# Results from evaluation

# Input image
img_resolution = [32, 32]			# Original DVS frame resolution
crop_top = 10						# Crop at the top
crop_bottom = 14					# Crop at the bottom
resolution = [16, 4]					# Resolution of reduced image

# Network parameters
sim_time = 50.0						# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.1				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max

# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min = 1500.						# Minimum initial random value
w0_max = 1501.						# Maximum initial random value
# These tau_n and tau_c parameters are suggested by Izhikevich, E.M. (2007). Solving the distal reward problem
# through linkage of STDP and dopamine signaling. Cereb. Cortex, 17(10), 2443-2452.
tau_n = 200.						# Time constant of reward signal
tau_c = 1000.						# Time constant of eligibility trace

reward_factor = 0.01				# Reward factor modulating reward signal strength
A_plus = 1.							# Constant scaling strength of potentiaion
A_minus = 1.						# Constant scaling strength of depression

# Snake turning model
n_max = sim_time//t_refrac          # Maximum input activity

r_min = 3.0							# Minimum turning radius
a_max = math.pi / 2					# Maximum turning angle

# Thermal Vision
default_temperature = 128			# Default temperature of the simulation

# Other
reset_steps = 5						# After how many steps without seeing the target should the simulation reset
rate = 20.							# ROS publication rate (step = 1/rate = 50ms)
training_length = 40000		    	# Length of training procedure (1 step ~ 50 ms)
evaluation_length = 20000			# Length of evaluation procedure

# Path numbers
plus_path = 2						# Simple path in + shape
plus_path_mirrored = 3
evaluation_path = 4					# Simple path in + shape
evaluation_path_mirrored = 5
