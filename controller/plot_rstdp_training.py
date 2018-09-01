#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import parameters as param
import argparse

# Configure Command Line interface
controller = dict(tf="target following controller", oa="obstacle avoidance controller")
parser = argparse.ArgumentParser(description='Plot the final weights and show it in a Window')
parser.add_argument('controller', choices=controller, default='oa', help="tf - target following, oa - obstacle avoidance")
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()

print "Using", controller[args.controller]
is_oa = args.controller == 'oa'

if is_oa:
	h5f = h5py.File(path.join(args.dir, param.training_file_oa), 'r')
else:
	h5f = h5py.File(path.join(args.dir, param.training_file_tf), 'r')
	w_tf = np.array(h5f['w_tf'], dtype=float)
	w_l = w_tf[:, 0]
	w_r = w_tf[:, 1]
	w_i = range(0, w_l.shape[0])



#w_p = np.zeros(10)#np.array(h5f['w_oa'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)
rewards = np.array(h5f['reward'], dtype=float)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)

xlim = w_r.shape[0]

fig = plt.figure(figsize=(7, 18))
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

"""
ax6 = plt.subplot(gs[5])
ax6.set_xlim((0, xlim))
ax6.set_ylim((param.w_min, param.w_max))
ax6.text(1000, 2800, 'Obstacle avoidance', color='0.4')
ax6.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
#for i in range(w_p.shape[1]):
#	for j in range(w_p.shape[2]):
#		plt.plot(w_i, w_p[:, i, j])
ax6.set_xlabel('Simulation Time [1 step = 50 ms]')
ax6.set_ylabel("Weight")
"""

fig.tight_layout()
plt.savefig(path.join(args.dir, "training_tf.png"))
if not args.noShow:
	plt.show()
