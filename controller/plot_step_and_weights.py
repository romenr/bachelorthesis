#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import parameters as param
import argparse
import pandas as pd

ewma = pd.stats.moments.ewma

# Configure Command Line interface
controller = dict(tf="target following controller", oa="obstacle avoidance controller")
parser = argparse.ArgumentParser(description='Plot the final weights and show it in a Window')
parser.add_argument('controller', choices=controller, default='oa', help="tf - target following, oa - obstacle avoidance")
parser.add_argument('-n', '--noShow', help='Do not show the resulting Plot in a window', action="store_true")
parser.add_argument('dir', help='Base directory of the experiment eg. ./data/session_xyz', default=param.default_dir)
args = parser.parse_args()

print "Using", controller[args.controller]
is_oa = args.controller == 'oa'

if is_oa:
	h5f = h5py.File(path.join(args.dir, param.training_file_oa), 'r')
	w = np.array(h5f['w_oa'], dtype=float)
	w_l = w[:, 0]
	w_r = w[:, 1]
	w_i = range(0, w_l.shape[0])
	dopamine = np.array(h5f['reward'], dtype=float)[:, 2]
else:
	h5f = h5py.File(path.join(args.dir, param.training_file_tf), 'r')
	w = np.array(h5f['w_tf'], dtype=float)
	w_l = w[:, 0]
	w_l = w_l.reshape(w_l.shape[0], -1)
	w_r = w[:, 1]
	w_r = w_r.reshape(w_r.shape[0], -1)
	w_i = range(0, w_l.shape[0])
	dopamine = np.array(h5f['reward'], dtype=float)[:, 0]

episode_steps = np.array(h5f["episode_steps"], dtype=float)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)

values_x = np.array(range(episode_steps.size))
success_y = episode_steps[episode_completed]
success_x = values_x[episode_completed]
failures_y = episode_steps[~episode_completed]
failures_x = values_x[~episode_completed]

# retrieve the dat
steps = np.array(h5f["episode_steps"], dtype=float)

# Plot
fig= plt.subplots(figsize=(9, 14))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

ax_1 = plt.subplot(411)
xlim1 = steps.size
ylim1 = steps.max(axis=0)*1.1
plt.plot(steps, lw=2, color='darkorange')
ax_1.set_xlim((0, xlim1))
ax_1.set_ylim((0, ylim1))
ax_1.set_ylabel('Time Steps')
ax_1.set_xlabel('Episode')
plt.grid()
plt.axhline(y=np.average(steps[steps > 400]), color='green', lw=3, linestyle='--')

for item in ([ax_1.title, ax_1.xaxis.label, ax_1.yaxis.label] + ax_1.get_xticklabels() + ax_1.get_yticklabels()):
	item.set_fontsize(16)

ax_1.scatter(success_x, success_y, marker='^', color='g', s=12)
ax_1.scatter(failures_x, failures_y, marker='x', color='r', s=12)

ax_2 = plt.subplot(412)
span_value = 20
time_step = np.arange(0, 40000)
fwd = ewma(dopamine, span=span_value)
bwd = ewma(dopamine[::-1], span=span_value)
c = np.vstack((fwd, bwd[::-1]))
c = np.mean(c, axis=0)
ax_2.set_ylabel('Dopamine Reward')
# ax_2.set_xlabel('Time Steps')
plt.plot(time_step, dopamine, lw=2, color='b', alpha=0.3)
plt.plot(time_step, c, lw=1, color='b')
for item in ([ax_2.title, ax_2.xaxis.label, ax_2.yaxis.label] + ax_2.get_xticklabels() + ax_2.get_yticklabels()):
	item.set_fontsize(16)

xlim = w_i[-1]
ymin1 = param.w_min
ymax1 = param.w_max
ax_3 = plt.subplot(413, sharex=ax_2)
# ax_3.set_title('Weights to left neuron', color='0.4')
ax_3.set_ylabel('Weight to Left Neuron')
ax_3.set_xlim((0,xlim))
ax_3.set_ylim((ymin1, ymax1))
plt.grid(True)
ax_3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
print w_l.shape
for i in range(w_l.shape[1]):
	plt.plot(w_i, w_l[:,i])
for item in ([ax_3.title, ax_3.xaxis.label, ax_3.yaxis.label] + ax_3.get_xticklabels() + ax_3.get_yticklabels()):
	item.set_fontsize(16)

ymin2 = param.w_min
ymax2 = param.w_max
ax_4 = plt.subplot(414, sharex=ax_3)
# ax_4.set_title('Weights to right neuron', color='0.4')
ax_4.set_ylabel('Weight to Right Neuron')
ax_4.set_xlim((0,xlim))
ax_4.set_ylim((ymin2,ymax2))
plt.grid(True)
ax_4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	plt.plot(w_i, w_r[:,i])
ax_4.set_xlabel('Simulation Time [1 step = 50 ms]')
for item in ([ax_4.title, ax_4.xaxis.label, ax_4.yaxis.label] + ax_4.get_xticklabels() + ax_4.get_yticklabels()):
	item.set_fontsize(16)

plt.grid()

plt.subplots_adjust(wspace=0., hspace=0.3, right=0.96, left=0.16, bottom=0.06, top=0.96)
if is_oa:
	plt.savefig(path.join(args.dir, "training_oa.pdf"), bbox_inches='tight')
else:
	plt.savefig(path.join(args.dir, "training_tf.pdf"), bbox_inches='tight')

plt.show()
