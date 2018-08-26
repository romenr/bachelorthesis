#!/usr/bin/env python

import h5py
import signal
import argparse
from os import path
import parameters as param
from environment import VrepEnvironment
import numpy as np

# Configure Command Line interface
controller = dict(tf="target following controller", oa="obstacle avoidance controller")
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('controller', choices=controller, default='oa', help="tf - target following, oa - obstacle avoidance")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()

# Import after argparse is done to prevent nest from showing up in help
from model import Model


# Stop the Simulation and save results up to that point
def signal_handler(signal, frame):
	global stop_signal_received
	stop_signal_received = True

stop_signal_received = False
signal.signal(signal.SIGINT, signal_handler)

print "Using", controller[args.controller]
is_oa = args.controller == controller['oa']

model = Model()
if is_oa:
	# Read tracking network weights
	h5f = h5py.File(path.join(args.dir, param.weights_file), 'r')
	w_tf = np.array(h5f['w_tf'], dtype=float)
	h5f.close()
	model.snn_tf.set_weights(w_tf[0], w_tf[1])

env = VrepEnvironment(param.plus_path, param.plus_path_mirrored)

# Arrays of variables that will be saved
weights_tf = []
weights_oa = []
rewards = []
angle_to_target = []
episode_steps = []
episode_completed = []

# Initialize environment, get state, get reward
s, r = env.reset()

for i in range(param.training_length):
	# Set rewards to the network that gets trained
	if is_oa:
		model.snn_oa.set_reward(r[2:])
	else:
		model.snn_tf.set_reward(r[:2])
	
	# Simulate network for 50 ms
	angle = model.simulate(s)

	# Feed output spikes into snake model
	# Get state, angle to target, reward, termination, step, path completed
	s, a, r, t, n, p = env.step(angle)

	if t:
		episode_steps.append(n)
		episode_completed.append(p)
	weights_tf.append(model.weights_tf)
	weights_oa.append(model.weights_oa)
	rewards.append(r)
	angle_to_target.append(a)

	# Print progress
	if i % (param.training_length/100) == 0:
		print "Training progress ", (i / (param.training_length/100)), "%"

	if stop_signal_received:
		break

# Save performance data
h5f = h5py.File(path.join(args.dir, param.training_file), 'w')
h5f.create_dataset('w_tf', data=weights_tf)
h5f.create_dataset('w_oa', data=weights_oa)
h5f.create_dataset('reward', data=rewards)
h5f.create_dataset('angle_to_target', data=angle_to_target)
h5f.create_dataset('episode_steps', data=episode_steps)
h5f.create_dataset('episode_completed', data=episode_completed)
h5f.close()

# Save trained weights
h5f = h5py.File(path.join(args.dir, param.weights_file), 'w')
h5f.create_dataset('w_tf', data=weights_tf[-1])
if is_oa:
	h5f.create_dataset('w_oa', data=weights_oa[-1])
h5f.close()

