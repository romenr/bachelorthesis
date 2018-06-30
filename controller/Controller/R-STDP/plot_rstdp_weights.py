#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
import parameters as param
import argparse

# Configure Command Line interface
parser = argparse.ArgumentParser(description='Plot Training progress and show it in a Window')
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('-f', '--inputFile', help="Input file", default='./data/rstdp_data.h5')
parser.add_argument('-o', '--outputFile', help="Output file")
args = parser.parse_args()

# R-STDP weights learned
# Fig. 5.7, Fig. 5.8, Fig. 5.10

h5f = h5py.File(args.inputFile, 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
weights_l = w_l[-1].T
weights_r = w_r[-1].T
print w_r.shape

fig = plt.figure(figsize=(6,6))

ax1 = plt.subplot(211)
plt.title('Left Weights')
plt.imshow(weights_l, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax1.text(i,j,int(label),ha='center',va='center')

ax2 = plt.subplot(212)
plt.title('Right Weights')
plt.imshow(weights_r, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax2.text(i,j,int(label),ha='center',va='center')

fig.tight_layout()
if args.outputFile is not None:
	plt.savefig(args.outputFile)
if not args.noShow:
	plt.show()
