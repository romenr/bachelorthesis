#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import parameters as param
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot the final weights and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()


h5f = h5py.File(path.join(args.dir, param.training_file_tf), 'r')

rewards = np.array(h5f['reward'], dtype=float)[:, 0]
episode_steps = np.array(h5f["episode_steps"], dtype=float)
episode_max = np.max(episode_steps)
for i in range(len(episode_steps) - 1):
	episode_steps[i + 1] += episode_steps[i]
rewards = np.split(rewards, episode_steps.astype(int))
rewards = rewards[1::2]
#rewards = rewards[0::2]

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)

# Skip s - 1 out of s values to increase plot readability
s = 1

# Plot 1 Plot Rewards in Episode i
ax1 = plt.subplot(gs[0, 0])
for i in range(len(rewards) - 2):
	ax1.plot(rewards[i][1::s], color=plt.cm.autumn(float(len(rewards) - i) / len(rewards)), label="Episode " + str(2 * i))
ax1.plot(rewards[-2][1::s], color=(0, 0, 1, 1), label="Last Episode")
ax1.plot([0, episode_max / float(s)], [0, 0], color='g', linestyle='-', linewidth=1)
ax1.set_ylabel("Reward")
ax1.set_xlabel("Step")

box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(path.join(args.dir, "rewards.png"))
if not args.noShow:
	plt.show()
