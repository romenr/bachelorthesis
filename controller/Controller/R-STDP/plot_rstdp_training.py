#!/usr/bin/env python

import numpy as np
import h5py
from environment import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
import parameters as param

# R-STDP training progress
# Fig. 5.6, Fig. 5.9

env = VrepEnvironment()

h5f = h5py.File(param.path + '/rstdp_data.h5', 'r')

xlim = 100000

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
e_o = np.array(h5f['e_o'], dtype=float)
e_i_o = np.array(h5f['e_i_o'], dtype=float)
e_i = np.array(h5f['e_i'], dtype=float)
e_i_i = np.array(h5f['e_i_i'], dtype=float)
rewards = np.array(h5f['reward'], dtype=float)
episode_steps = np.array(h5f["episode_steps"], dtype=float)

fig = plt.figure(figsize=(7, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])

ax1 = plt.subplot(gs[0])
plt.plot(episode_steps)
ax1.set_ylabel('Time steps')
ax1.set_xlabel('Episode')

ax3 = plt.subplot(gs[1])
ax3.set_xlim((0, xlim))
ax3.set_ylim((0, 3000))
ax3.set_xticklabels([])
ax3.text(1000, 2800, 'Left Motor', color='0.4')
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:, i, j])
ax3.set_xlabel('Simulation Time [1 step = 50 ms]')
ax3.set_ylabel("Weight")

ax4 = plt.subplot(gs[2])
ax4.set_xlim((0, xlim))
ax4.set_ylim((0, 3000))
ax4.text(1000, 2800, 'Right Motor', color='0.4')
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	for j in range(w_r.shape[2]):
		plt.plot(w_i, w_r[:, i, j])
ax4.set_xlabel('Simulation Time [1 step = 50 ms]')
ax4.set_ylabel("Weight")


fig.tight_layout()
plt.show()
