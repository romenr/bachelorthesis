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
parser.add_argument('-s', '--step', help='Show the weights at step s. Default last step -1', default=-1)
args = parser.parse_args()

weight_index = int(args.step)


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
w_h = np.array(h5f['w_h'], dtype=float)
# weights_l = w_l[weight_index].T.reshape(param.hidden_layer_size / 5, 5)
# weights_r = w_r[weight_index].T.reshape(param.hidden_layer_size / 5, 5)
weights_h = w_h[weight_index]

fig = plt.figure(figsize=(30, 7))
gs = gridspec.GridSpec(1 + weights_h.shape[0] / 2, 2)

ax1 = plt.subplot(gs[0, 0])
plt.title('Left Weights')
ax1.set_ylabel('Step '+str(w_r.shape[0]))
# plot_weights(ax1, weights_l)
print w_l[-1]
print w_r[-1]

ax2 = plt.subplot(gs[0, 1])
plt.title('Right Weights')
# plot_weights(ax2, weights_r)

plt.title('Hidden weights')
for i in range(weights_h.shape[0]):
	ax = plt.subplot(gs[1 + i // 2, i % 2])
	plot_weights(ax, weights_h[i].reshape(param.resolution).T)

fig.tight_layout()
plt.savefig(path.join(args.dir, "weights.png"))
if not args.noShow:
	plt.show()
