#!/usr/bin/env python

import h5py
import signal
import argparse
from os import path
import parameters as param
from snn import SpikingNeuralNetwork
from environment import VrepEnvironment
import numpy as np

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
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
episode_completed = []

# Initialize environment, get state, get reward
s, r = env.reset()

for i in range(param.training_length):
	
	# Simulate network for 50 ms
	# Get left and right output spikes, get weights
	reward = np.array([-r, r]) * param.reward_factor
	snn.set_reward(reward)
	n_l, n_r, w_l, w_r = snn.simulate(s)

	# Feed output spikes into snake model
	# Get state, angle to target, reward, termination, step, path completed
	s, a, r, t, n, p = env.step(n_l, n_r)

	if t:
		episode_steps.append(n)
		episode_completed.append(p)
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
h5f = h5py.File(path.join(args.dir, param.training_file), 'w')
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
h5f.create_dataset('episode_completed', data=episode_completed)
h5f.close()

# Save trained weights
h5f = h5py.File(path.join(args.dir, param.weights_file), 'w')
h5f.create_dataset('w_l', data=weights_l[-1])
h5f.create_dataset('w_r', data=weights_r[-1])
h5f.close()

