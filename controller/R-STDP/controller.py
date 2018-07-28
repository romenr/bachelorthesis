#!/usr/bin/env python

import h5py
import signal
import argparse
import numpy as np
from network import SpikingNeuralNetwork
from environment import VrepEnvironment
import parameters as param

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Run the model')
parser.add_argument('-n', '--noShow', help='Do not show training information in additional window', action="store_true")
parser.add_argument('-f', '--inputFile', help="Input file", default='./data/rstdp_data.h5')
parser.add_argument('-o', '--outputFile', help="Output file", default='./data/controller_data.h5')
args = parser.parse_args()


# Stop the Simulation and save results up to that point
def signal_handler(signal, frame):
	global stop_signal_received
	stop_signal_received = True

stop_signal_received = False
signal.signal(signal.SIGINT, signal_handler)

# Read network weights
h5f = h5py.File(args.inputFile, 'r')
w_l = np.array(h5f['w_l'], dtype=float)[-1]
w_r = np.array(h5f['w_r'], dtype=float)[-1]
h5f.close()

snn = SpikingNeuralNetwork()
env = VrepEnvironment(param.evaluation_path, param.evaluation_path_mirrored)
snn.set_weights(w_l, w_r)

# Arrays of variables that will be saved
reward = []
angle_to_target = []
episode_steps = []

# Initialize environment, get state, get reward
s, r = env.reset()

for i in range(param.evaluation_length):

	# Simulate network for 50 ms
	# Get left and right output spikes, get weights
	# Fix the Reward at 0 to prevent the network from changing
	n_l, n_r, w_l, w_r = snn.simulate(s, 0.)

	# Feed output spikes into snake model
	# Get state, angle to target, reward, termination, step
	s, a, r, t, n = env.step(n_l, n_r)

	# Store information that should be saved
	if t:
		episode_steps.append(n)
	reward.append(r)
	angle_to_target.append(a)

	# Print progress
	if i % (param.evaluation_length / 100) == 0:
		print "Evaluation progress ", (i / (param.evaluation_length / 100)), "%"

	if stop_signal_received:
		break

# Save performance data
h5f = h5py.File(args.outputFile, 'w')
h5f.create_dataset('angle_to_target', data=angle_to_target)
h5f.create_dataset('reward', data=reward)
h5f.create_dataset('episode_steps', data=episode_steps)
h5f.close()
