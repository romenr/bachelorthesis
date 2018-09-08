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

data_x = [895.55, 976.19, 1316.7, 1261.87, 1229.12]
data_y = [0.01, 0.005, 0.002, 0.001, 0.0005]

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)

ax1 = plt.subplot(gs[0, 0])

ax1.plot(data_y, data_x)
ax1.set_xlabel("Learning rate")
ax1.set_ylabel("Error sum")

fig.tight_layout()
plt.savefig(path.join(args.dir, "learning_rate.png"))
if not args.noShow:
	plt.show()
