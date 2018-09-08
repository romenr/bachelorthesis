import numpy as np
import h5py
from environment import *
from parameters import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

env = VrepEnvironment()

h5f = h5py.File(path + '/rstdp_performance_data_' + track_type +'.h5', 'r')
distances = np.array(h5f['distance'], dtype=float)

# Plot
fig = plt.subplots(figsize=(12, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1.6])

ax_1 = plt.subplot(gs[0])

xlim1 = distances.size
# ylim1 = 1.1
plt.plot(distances, lw=3, color='darkorange')
ax_1.set_xlim((0, xlim1))
ax_1.set_ylim((-1.2, 1.2))
ax_1.set_ylabel('Distance')
ax_1.set_xlabel('Time Steps')
my_xtick = np.arange(0, len(distances), 500)
ax_1.set_xticks(my_xtick)
plt.grid()
plt.axhline(y=-1, color='green', lw=3, linestyle='--')
plt.axhline(y=1, color='green', lw=3, linestyle='--')

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
plt.savefig(path + '/plot_rstdp_performance_'+ track_type +'.pdf', bbox_inches='tight')
plt.show()