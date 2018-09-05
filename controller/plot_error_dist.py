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

h5f = None
if is_oa:
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_oa), 'r')
else:
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_tf), 'r')

rewards = np.array(h5f['reward'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=int)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)

# Calculate Values
angle_to_target_sum = [0]
for i in range(angle_to_target.size):
	angle_to_target_sum.append(angle_to_target_sum[i-1] + abs(angle_to_target[i]))

# Plot 5 Distribution of Distance
ax5 = plt.subplot(gs[0])
ax5.hist(angle_to_target, 50, facecolor='b', alpha=0.75)
ax5.set_ylabel("Frequency")
ax5.set_xlabel("Distance error")
ax5.grid(True)
#ax5.text(0.1, 0.9, 'mean = '+str(np.mean(angle_to_target))+' variance = '+str(np.var(angle_to_target)), transform=ax5.transAxes)
print 'mean = ', str(np.mean(angle_to_target)), ' variance = ', str(np.var(angle_to_target))

fig.tight_layout()
plt.savefig(path.join(args.dir, "eval_tf.png"))
if not args.noShow:
	plt.show()
