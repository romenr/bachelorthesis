#!/usr/bin/env python

import h5py
import signal
import argparse
import parameters as param
from network import SpikingNeuralNetwork
from environment import VrepEnvironment

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-n', '--noShow', help='Do not show training information in additional window', action="store_true")
parser.add_argument('-o', '--outputFile', help="Output file", default='./data/rstdp_data.h5')
args = parser.parse_args()


# Stop the Simulation and save results up to that point
def signal_handler(signal, frame):
	global stop_signal_received
	stop_signal_received = True

stop_signal_received = False
signal.signal(signal.SIGINT, signal_handler)

snn = SpikingNeuralNetwork()
env = VrepEnvironment(param.plus_path, param.plus_path_mirrored)

# Arrays of variables that will be saved
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

# Initialize environment, get state, get reward
s, r = env.reset()

for i in range(param.training_length):
	
	# Simulate network for 50 ms
	# Get left and right output spikes, get weights
	n_l, n_r, w_l, w_r = snn.simulate(s, r)

	# Feed output spikes into snake model
	# Get state, angle to target, reward, termination, step
	s, a, r, t, n = env.step(n_l, n_r)

	if t:
		episode_steps.append(n)
	weights_l.append(w_l)
	weights_r.append(w_r)
	weights_i.append(i)
	rewards.append(r)
	angle_to_target.append(a)

	# Print progress
	if i % (param.training_length/100) == 0:
		print "Training progress ", (i / (param.training_length/100)), "%"

	if stop_signal_received:
		break

# Save performance data
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
