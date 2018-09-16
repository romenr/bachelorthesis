import numpy as np
import h5py
from environment import *
from os import path
import parameters as param
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_oa), 'r')
else:
	h5f = h5py.File(path.join(args.dir, param.evaluation_file_tf), 'r')


episode_steps = np.array(h5f["episode_steps"], dtype=int)
episode_completed = np.array(h5f['episode_completed'], dtype=bool)
distances = np.array(h5f['angle_to_target'], dtype=float)
#distances = np.array(h5f["target_pos"], dtype=float)

distances = distances[episode_steps[0]:np.sum(episode_steps[:2])]

# Plot
fig = plt.subplots(figsize=(12, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1.6])

ax_1 = plt.subplot(gs[0])

xlim1 = distances.size
# ylim1 = 1.1
plt.plot(distances, lw=3, color='darkorange')
ax_1.set_xlim((0, xlim1))
ax_1.set_ylim((-1.2, 1.2))
ax_1.set_ylabel('Angle')
ax_1.set_xlabel('Time Steps')
my_xtick = np.arange(0, len(distances), 500)
ax_1.set_xticks(my_xtick)
plt.grid()
plt.axhline(y=-0.52, color='green', lw=3, linestyle='--')
plt.axhline(y=0.52, color='green', lw=3, linestyle='--')

# Plot Histgram figure
ax_2 = plt.subplot(gs[1])
b = [x*0.01 for x in range(-100, 100)]
plt.hist(distances, bins=b, normed=True, facecolor='g', edgecolor='g', alpha=1, linewidth=4, orientation=u'horizontal')
ax_2.set_xticklabels([])
ax_2.set_yticklabels([])
ax_2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
ax_2.set_title('R-STDP \ne = '+str('{:4.3f}'.format(abs(distances).mean())), loc='left', size='16', position=(1.1,0.2))
plt.axhline(y=0, linewidth=0.5, color='0.')
ax_2.set_xlabel('Histogram')

for item in ([ax_1.title, ax_1.xaxis.label, ax_1.yaxis.label] +
				ax_1.get_xticklabels() + ax_1.get_yticklabels()):
	item.set_fontsize(20)

for item in ([ax_2.title, ax_2.xaxis.label, ax_2.yaxis.label] +
				ax_2.get_xticklabels() + ax_2.get_yticklabels()):
	item.set_fontsize(20)

plt.subplots_adjust(wspace=0., hspace=0.1, right=0.88, left=0.1, bottom=0.2)
if is_oa:
	plt.savefig(path.join(args.dir, "performance_oa.pdf"), bbox_inches='tight')
else:
	plt.savefig(path.join(args.dir, "performance_tf.pdf"), bbox_inches='tight')

plt.show()