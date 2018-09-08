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
	w_tf = np.array(h5f['w_tf'], dtype=float)
	w_l = w_tf[:, 0]
	w_r = w_tf[:, 1]
	w_i = range(0, w_l.shape[0])
	w_p = np.array(h5f['w_oa'], dtype=float)
else:
	h5f = h5py.File(path.join(args.dir, param.training_file_tf), 'r')
	w_tf = np.array(h5f['w_tf'], dtype=float)
	w_l = w_tf[:, 0]
	w_r = w_tf[:, 1]
	w_i = range(0, w_l.shape[0])

episode_steps = np.array(h5f["episode_steps"], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)
rewards = np.array(h5f['reward'], dtype=float)
angle_to_target = np.array(h5f['angle_to_target'], dtype=float)

xlim = w_r.shape[0]

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)

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

fig.tight_layout()
if is_oa:
	plt.savefig(path.join(args.dir, "success_oa.png"))
else:
	plt.savefig(path.join(args.dir, "success_tf.png"))
if not args.noShow:
	plt.show()
