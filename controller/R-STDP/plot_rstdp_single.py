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
distance = np.array(h5f['distance'], dtype=float)

xlim = w_r.shape[0]

fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(1, 1)

ax1 = plt.subplot(gs[0])
ax1.set_xlim((0, xlim))
ax1.set_ylim((param.w_min, param.w_max))
ax1.set_xticklabels([])
ax1.text(1000, 2800, 'Left Motor', color='0.4')
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:, i, j])
ax1.set_xlabel('Simulation Time [1 step = 50 ms]')
ax1.set_ylabel("Weight")

fig.tight_layout()
if args.outputFile is not None:
	plt.savefig(args.outputFile)
if not args.noShow:
	plt.show()
