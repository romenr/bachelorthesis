#!/usr/bin/env python

from network import *
from environment import *
from parameters import *
import h5py
import signal
import sys
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-n', '--noShow', help='Do not show training information in additional window', action="store_true")
parser.add_argument('-o', '--outputFile', help="Output file", default='./data/rstdp_data.h5')
args = parser.parse_args()


def signal_handler(signal, frame):
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

snn = SpikingNeuralNetwork()
env = VrepEnvironment(plus_path, plus_path_mirrored)

# Variables that will be saved
weights_r = []
weights_l = []
weights_i = []
episode_position_o = []
episode_i_o = []
episode_position_i = []
episode_i_i = []
rewards = []
angle_to_target = []
episode_steps = []

# Initialize environment, get initial state, initial reward
s, r = env.reset()

for i in range(training_length):
	
	# Simulate network for 50 ms
	# get number of output spikes and network weights
	n_l, n_r, w_l, w_r = snn.simulate(s, r)

	# Feed output spikes into snake model
	# Get state, angle to target, reward, termination, step
	s, a, r, t, n = env.step(n_l, n_r)

	if t:
		episode_steps.append(n)

	# Save weights every simulation step
	weights_l.append(w_l)
	weights_r.append(w_r)
	weights_i.append(i)
	rewards.append(r)
	angle_to_target.append(a)

	if i % (training_length/100) == 0:
		print "Training progress ", (i / (training_length/100)), "%"

# Save data
h5f = h5py.File(args.outputFile, 'w')
h5f.create_dataset('w_l', data=weights_l)
h5f.create_dataset('w_r', data=weights_r)
h5f.create_dataset('w_i', data=weights_i)
h5f.create_dataset('e_o', data=episode_position_o)
h5f.create_dataset('e_i_o', data=episode_i_o)
h5f.create_dataset('e_i', data=episode_position_i)
h5f.create_dataset('e_i_i', data=episode_i_i)
h5f.create_dataset('reward', data=rewards)
h5f.create_dataset('angle_to_target', data=angle_to_target)
h5f.create_dataset('episode_steps', data=episode_steps)
h5f.close()
