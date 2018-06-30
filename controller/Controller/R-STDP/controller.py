#!/usr/bin/env python

from network import *
from environment import *
from parameters import *
import h5py
import signal
import sys
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Run the model')
parser.add_argument('-n', '--noShow', help='Do not show training information in additional window', action="store_true")
parser.add_argument('-f', '--inputFile', help="Input file", default='./data/rstdp_data.h5')
parser.add_argument('-o', '--outputFile', help="Output file", default='./data/controller_data.h5')
args = parser.parse_args()


def signal_handler(signal, frame):
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

snn = SpikingNeuralNetwork()
env = VrepEnvironment()

# Read network weights
h5f = h5py.File(args.inputFile, 'r')
w_l = np.array(h5f['w_l'], dtype=float)[-1]
w_r = np.array(h5f['w_r'], dtype=float)[-1]
# Set network weights
snn.set_weights(w_l, w_r)
h5f.close()

# Variables that will be saved
reward = []
distance = []
episode_steps = []

# Initialize environment, get state, get reward
s, r = env.reset()

for i in range(evaluation_length):

	# Simulate network for 50 ms
	# Get left and right output spikes, get weights
	n_l, n_r, w_l, w_r = snn.simulate(s,r)

	# Feed output spikes into steering wheel model
	# Get state, distance, position, reward, termination, step
	s, d, r, t, n = env.step(n_l, n_r)

	if t:
		episode_steps.append(n)

	# Store position, distance
	reward.append(r)
	distance.append(d)

	if i % (training_length / 100) == 0:
		print "Evaluation progress ", (i / (training_length / 100)), "%"

# Save performance data
h5f = h5py.File(args.outputFile, 'w')
h5f.create_dataset('distance', data=distance)
h5f.create_dataset('reward', data=reward)
h5f.create_dataset('episode_steps', data=episode_steps)
h5f.close()
