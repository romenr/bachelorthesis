import numpy as np
import h5py
from environment import *
from os import path
import parameters as param
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import linregress

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
distances_vision = np.array(h5f["target_pos"], dtype=float)

distances = (distances[episode_steps[0]:np.sum(episode_steps[:2])])
distances_vision = (distances_vision[episode_steps[0]-1:np.sum(episode_steps[:2])-1])
print linregress(distances, distances_vision)

# Plot
fig = plt.subplots()
gs = gridspec.GridSpec(1, 1)

ax_1 = plt.subplot(gs[0])

plt.scatter(distances_vision, distances)
ax_1.set_ylabel('Distance [rad]')
ax_1.set_xlabel('Normalized centroid position')
my_xtick = np.arange(0, len(distances), 500)
plt.grid()


plt.subplots_adjust(wspace=0., hspace=0.1, right=0.88, left=0.1, bottom=0.2)
if is_oa:
	plt.savefig(path.join(args.dir, "performance_oa.pdf"), bbox_inches='tight')
else:
	plt.savefig(path.join(args.dir, "performance_tf.pdf"), bbox_inches='tight')

plt.show()