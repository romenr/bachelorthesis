#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
import parameters as param
from matplotlib import gridspec
from os import path
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot Training progress and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()


def plot_weights(ax, weights):
	ax.imshow(weights.astype(int), alpha=0.5)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	ax.set_yticks([])
	ax.set_xticks([])
	for (j, i), label in np.ndenumerate(weights):
		ax.text(i, j, int(label), ha='center', va='center')

h5f = h5py.File(path.join(args.dir, param.training_file), 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
m = w_r.shape[0] / 2
weights_l0 = w_l[0].T
weights_r0 = w_r[0].T
weights_l1 = w_l[m].T
weights_r1 = w_r[m].T
weights_l2 = w_l[-1].T
weights_r2 = w_r[-1].T

fig = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(3, 2)

ax1 = plt.subplot(gs[0, 0])
plt.title('Left Weights')
ax1.set_ylabel('Step 0')
plot_weights(ax1, weights_l0)

ax2 = plt.subplot(gs[0, 1])
plt.title('Right Weights')
plot_weights(ax2, weights_r0)

ax3 = plt.subplot(gs[1, 0])
ax3.set_ylabel('Step '+str(m))
plot_weights(ax3, weights_l1)

ax4 = plt.subplot(gs[1, 1])
plot_weights(ax4, weights_r1)

ax5 = plt.subplot(gs[2, 0])
ax5.set_ylabel('Step '+str(w_r.shape[0]))
plot_weights(ax5, weights_l2)

ax6 = plt.subplot(gs[2, 1])
plot_weights(ax6, weights_r2)

fig.tight_layout()
plt.savefig(path.join(args.dir, "weights.png"))
if not args.noShow:
	plt.show()
