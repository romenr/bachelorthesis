#!/usr/bin/env python

import numpy as np
from network import *
from environment import *
from parameters import *
import h5py
import signal
import sys

def signal_handler(signal, frame):
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

snn = SpikingNeuralNetwork()
env = VrepEnvironment()
weights_r = []
weights_l = []
weights_i = []
episode_position_o = []
episode_i_o = []
episode_position_i = []
episode_i_i = []
rewards = []

# Initialize environment, get initial state, initial reward
s,r = env.reset()

for i in range(training_length):
	
	# Simulate network for 50 ms
	# get number of output spikes and network weights
	n_l, n_r, w_l, w_r = snn.simulate(s,r)

	# Feed output spikes in steering wheel model
	# Get state, distance, reward, termination, step, lane
	s,d,r,t,n,o = env.step(n_l, n_r)

	rewards.append(r)

	# Save weights every 100 simulation steps
	if i % 100 == 0:
		weights_l.append(w_l)
		weights_r.append(w_r)
		weights_i.append(i)

	# Save last position if episode is terminated
	#if t:
	#	if o:
	#		episode_position_o.append(p)
	#		episode_i_o.append(i)
	#	else:
	#		episode_position_i.append(p)
	#		episode_i_i.append(i)
	#	print i, p

	if i % (training_length/100) == 0:
		print "Training progress ", (i / (training_length/100)), "%"

# Save data
h5f = h5py.File(path + '/rstdp_data.h5', 'w')
h5f.create_dataset('w_l', data=weights_l)
h5f.create_dataset('w_r', data=weights_r)
h5f.create_dataset('w_i', data=weights_i)
h5f.create_dataset('e_o', data=episode_position_o)
h5f.create_dataset('e_i_o', data=episode_i_o)
h5f.create_dataset('e_i', data=episode_position_i)
h5f.create_dataset('e_i_i', data=episode_i_i)
h5f.create_dataset('reward', data=rewards)
h5f.close()
