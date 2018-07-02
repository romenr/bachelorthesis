#!/usr/bin/env python

import math

# Input image
img_resolution = [32, 32]			# Original DVS frame resolution
crop_top = 14						# Crop at the top
crop_bottom = 14					# Crop at the bottom
resolution = [8, 4]					# Resolution of reduced image

# Network parameters
sim_time = 50.0						# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.1				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max
max_spikes = 15.					# number of events during each step for maximum poisson frequency

# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min = 1500.						# Minimum initial random value
w0_max = 1501.						# Maximum initial random value
# These tau_n and tau_c parameters are suggested by Izhikevich, E.M. (2007). Solving the distal reward problem
# through linkage of STDP and dopamine signaling. Cereb. Cortex, 17(10), 2443-2452.
tau_n = 200.						# Time constant of reward signal
# Reducing tau_c reduces the variance in the training significantly
tau_c = 1000.						# Time constant of eligibility trace

reward_factor = 0.00125				# Reward factor modulating reward signal strength
# Reducing these constants reduces the variance in the training significantly
A_plus = 1.						# Constant scaling strength of potentiaion
A_minus = 1.						# Constant scaling strength of depression

# Snake turning model
n_max = sim_time//t_refrac          # Maximum input activity

r_min = 3.0							# Minimum turning radius
a_max = math.pi/2					# Maximum turning angle

# Thermal Vision
default_temperature = 128			# Default temperature of the simulation

# Other
reset_distance = 0.2				# Reset distance
rate = 20.							# ROS publication rate motor speed
training_length = 40000		    # Length of training procedure (1 step ~ 50 ms)
evaluation_length = 20000		# Length of evaluation procedure
trial_step_max = 2000				# Maximum number of Steps in one Trial
