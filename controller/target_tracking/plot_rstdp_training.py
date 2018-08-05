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

h5f = h5py.File(path.join(args.dir, param.training_file), 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)
rewards = np.array(h5f['reward'], dtype=float)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)

xlim = w_r.shape[0]

fig = plt.figure(figsize=(7, 12))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 2, 2, 1, 1])

ax1 = plt.subplot(gs[0])
values_x = np.array(range(episode_steps.size))
success_y = episode_steps[episode_completed]
success_x = values_x[episode_completed]
failures_y = episode_steps[~episode_completed]
failures_x = values_x[~episode_completed]
ax1.scatter(success_x, success_y, marker='^', color='g')
ax1.scatter(failures_x, failures_y, marker='x', color='r')
ax1.set_ylabel("Duration")
ax1.set_xlabel("Episode")

ax2 = plt.subplot(gs[1])
ax2.set_xlim((0, xlim))
ax2.set_ylim((param.w_min, param.w_max))
ax2.set_xticklabels([])
ax2.text(1000, 2800, 'Left Motor', color='0.4')
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	plt.plot(w_i, w_l[:, i])
ax2.set_xlabel('Simulation Time [1 step = 50 ms]')
ax2.set_ylabel("Weight")

ax3 = plt.subplot(gs[2])
ax3.set_xlim((0, xlim))
ax3.set_ylim((param.w_min, param.w_max))
ax3.text(1000, 2800, 'Right Motor', color='0.4')
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	plt.plot(w_i, w_l[:, i])
ax3.set_xlabel('Simulation Time [1 step = 50 ms]')
ax3.set_ylabel("Weight")

# Plot 4 Plot Reward at each Step
ax4 = plt.subplot(gs[3])
ax4.plot(rewards)
ax4.set_ylabel("Reward right neuron")
ax4.set_xlabel("Step")

# Plot 5 Plot Distance between Car and camera center at each Step
ax5 = plt.subplot(gs[4])
ax5.plot(angle_to_target)
ax5.set_ylabel("Angle error")
ax5.set_xlabel("Step")


fig.tight_layout()
plt.savefig(path.join(args.dir, "training.png"))
if not args.noShow:
	plt.show()
