#!/usr/bin/env python

import numpy as np
import numpy as np
import h5py
from environment import *
from os import path
import parameters as param
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot the final weights and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()

h5f = h5py.File(path.join(args.dir, param.training_file_tf), 'r')

w_tf = np.array(h5f['w_tf'], dtype=float)
w_i = range(0, w_tf.shape[1])
print w_tf.shape


episode_steps = np.array(h5f["episode_steps"], dtype=int)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)

failures = episode_steps[~episode_completed]
success = np.sum(failures) + episode_steps[episode_completed][0]
print success
# success = -1
weights_l = np.swapaxes(w_tf[success][0], 0, 1)
weights_r = np.swapaxes(w_tf[success][1], 0, 1)


fig = plt.figure(figsize=(9, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

ax3 = plt.subplot(211)
plt.title('Left Neuron Weights')
plt.imshow(weights_l, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax3.text(i,j,int(label),ha='center',va='center', fontsize=8)


ax4 = plt.subplot(212)
plt.title('Right Neuron Weights')
plt.imshow(weights_r, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax4.text(i,j,int(label),ha='center',va='center', fontsize=8)


for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
			ax3.get_xticklabels() + ax3.get_yticklabels()):
	item.set_fontsize(20)

for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
			ax4.get_xticklabels() + ax4.get_yticklabels()):
	item.set_fontsize(20)

plt.subplots_adjust(wspace=0., hspace=0., right=0.96, left=0.16, bottom=0.06, top=0.94)
plt.savefig(path.join(args.dir, "weights_2000.pdf"), bbox_inches='tight')
plt.show()
