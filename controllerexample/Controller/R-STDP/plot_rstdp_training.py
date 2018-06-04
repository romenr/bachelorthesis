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

fig = plt.figure(figsize=(7,8))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 2]) 

ax1 = plt.subplot(gs[0])
plt.plot(rewards)
ax1.set_ylabel('Reward')
ax1.set_xlabel('episode')

ax2 = plt.subplot(gs[1])
ax2.set_xlim((0,xlim))
ax2.set_xticklabels([])
ax2.text(1000, 25, 'Inner Lane', color='0.4')
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(e_i_i,e_i, 'x', markersize=10.)
ax6 = ax2.twinx()
ax6.set_ylabel('Section')
ax6.set_yticklabels(['C','B','A','F','E','D'])
ax6.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=False, right=False)


ax3 = plt.subplot(gs[2])
ax3.set_ylabel('Weight', position=(0.,0.))
ax3.set_xlim((0,xlim))
ax3.set_ylim((0,2300))
ax3.set_xticklabels([])
ax3.text(1000, 2100, 'Left Motor', color='0.4')
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:,i,j])

ax4 = plt.subplot(gs[3], sharey=ax3)
ax4.set_xlim((0,xlim))
ax4.set_ylim((0,2300))
ax4.text(1000, 2100, 'Right Motor', color='0.4')
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	for j in range(w_r.shape[2]):
		plt.plot(w_i, w_r[:,i,j])
ax4.set_xlabel('Simulation Time [1 step = 50 ms]')


fig.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.)
plt.show()
