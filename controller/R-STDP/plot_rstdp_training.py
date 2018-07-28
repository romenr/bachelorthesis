#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
import parameters as param
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot the final weights and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('-f', '--inputFile', help="Input file", default='./data/rstdp_data.h5')
parser.add_argument('-o', '--outputFile', help="Output file")
args = parser.parse_args()

h5f = h5py.File(args.inputFile, 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=float)
rewards = np.array(h5f['reward'], dtype=float)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)

xlim = w_r.shape[0]

fig = plt.figure(figsize=(7, 12))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 2, 2, 1, 1])

ax1 = plt.subplot(gs[0])
ax1.plot(episode_steps)
ax1.set_ylabel('Time steps')
ax1.set_xlabel('Episode')

ax2 = plt.subplot(gs[1])
ax2.set_xlim((0, xlim))
ax2.set_ylim((param.w_min, param.w_max))
ax2.set_xticklabels([])
ax2.text(1000, 2800, 'Left Motor', color='0.4')
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:, i, j])
ax2.set_xlabel('Simulation Time [1 step = 50 ms]')
ax2.set_ylabel("Weight")

ax3 = plt.subplot(gs[2])
ax3.set_xlim((0, xlim))
ax3.set_ylim((param.w_min, param.w_max))
ax3.text(1000, 2800, 'Right Motor', color='0.4')
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	for j in range(w_r.shape[2]):
		plt.plot(w_i, w_r[:, i, j])
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
if args.outputFile is not None:
	plt.savefig(args.outputFile)
if not args.noShow:
	plt.show()
