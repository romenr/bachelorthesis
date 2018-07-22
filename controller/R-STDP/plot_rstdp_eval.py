#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import parameters as param

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot the Controller evaluation results and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('-f', '--inputFile', help="Input file", default='./data/controller_data.h5')
parser.add_argument('-o', '--outputFile', help="Output file")
args = parser.parse_args()

h5f = h5py.File(args.inputFile, 'r')

rewards = np.array(h5f['reward'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=int)
distance = np.array(h5f['distance'], dtype=float)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(3, 1)

# Calculate Values
distance_sum = [0]
for i in range(distance.size):
	distance_sum.append(distance_sum[i-1] + abs(distance[i]))

# Plot 3 Plot Distance between Car and camera center at each Step
ax3 = plt.subplot(gs[0, 0])
ax3.plot(distance)
ax3.set_ylabel("Distance error")
ax3.set_xlabel("Step")

# Plot 5 Distribution of Distance
ax5 = plt.subplot(gs[1:, 0])
ax5.hist(distance, 50, facecolor='b', alpha=0.75)
ax5.set_ylabel("Frequency")
ax5.set_xlabel("Distance error")
ax5.grid(True)
ax5.text(0.1, 0.9, 'mean = '+str(np.mean(distance))+' variance = '+str(np.var(distance)), transform=ax5.transAxes)

fig.tight_layout()
if args.outputFile is not None:
	plt.savefig(args.outputFile)
if not args.noShow:
	plt.show()
