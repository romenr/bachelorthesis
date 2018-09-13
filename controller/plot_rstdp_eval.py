#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import argparse
import parameters as param

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
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_oa), 'r')
else:
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_tf), 'r')

rewards = np.array(h5f['reward'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=int)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(3, 2)

# Calculate Values
angle_to_target_sum = [0]
for i in range(angle_to_target.size):
	angle_to_target_sum.append(angle_to_target_sum[i-1] + abs(angle_to_target[i]))


# Plot 1 Plot Steps in Episode i
ax1 = plt.subplot(gs[0, 0])
values_x = np.array(range(episode_steps.size))
success_y = episode_steps[episode_completed]
success_x = values_x[episode_completed]
failures_y = episode_steps[~episode_completed]
failures_x = values_x[~episode_completed]
ax1.scatter(success_x, success_y, marker='^', color='g')
ax1.scatter(failures_x, failures_y, marker='x', color='r')
ax1.set_ylabel("Duration")
ax1.set_xlabel("Episode")

# Plot 2 Plot Reward at each Step
ax2 = plt.subplot(gs[1, 0])
ax2.plot(rewards)
ax2.set_ylabel("Reward right neuron")
ax2.set_xlabel("Step")

# Plot 3 Plot Distance between Car and camera center at each Step
ax3 = plt.subplot(gs[2, 0])
ax3.plot(angle_to_target)
ax3.set_ylabel("Distance error")
ax3.set_xlabel("Step")

# Plot 4 Plot Sum Distance between Car and camera center at each Step
ax4 = plt.subplot(gs[0, 1])
ax4.plot(angle_to_target_sum)
ax4.set_ylabel("Absolute distance error sum")
ax4.set_xlabel("Step")
ax4.text(0.1, 0.9, "sum = " + str(angle_to_target_sum[-1]), transform=ax4.transAxes)

# Plot 5 Distribution of Distance
ax5 = plt.subplot(gs[1:, 1])
ax5.hist(angle_to_target, 50, facecolor='b', alpha=0.75)
ax5.set_ylabel("Frequency")
ax5.set_xlabel("Distance error")
ax5.grid(True)
ax5.text(0.1, 0.9, 'mean = '+str(np.mean(angle_to_target))+' variance = '+str(np.var(angle_to_target)), transform=ax5.transAxes)

fig.tight_layout()
if is_oa:
	plt.savefig(path.join(args.dir, "eval_oa.pdf"), bbox_inches='tight')
else:
	plt.savefig(path.join(args.dir, "eval_tf.pdf"), bbox_inches='tight')
if not args.noShow:
	plt.show()
