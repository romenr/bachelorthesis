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
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()


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

fig = plt.figure(figsize=(9, 6))
gs = {}
gs = gridspec.GridSpec(1, 1, height_ratios=[2])

ax2 = plt.subplot(gs[0])
ax2.set_xlim((0, xlim))
ax2.set_ylim((param.w_min, param.w_max))
ax2.text(1000, 2800, 'Left Motor', color='0.4')
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:, i, j])
ax2.set_xlabel('Simulation Time [1 step = 50 ms]')
ax2.set_ylabel("Weight")


fig.tight_layout()
plt.savefig(path.join(args.dir, "weights_tf.png"))
if not args.noShow:
	plt.show()
