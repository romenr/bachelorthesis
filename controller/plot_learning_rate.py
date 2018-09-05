#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import argparse
import parameters as param

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot the Controller evaluation results and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()

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

fig.tight_layout()
plt.savefig(path.join(args.dir, "learning_rate.png"))
if not args.noShow:
	plt.show()
