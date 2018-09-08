#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from parameters import *
from matplotlib import gridspec


h5f = h5py.File(path + '/rstdp_turn_data.h5', 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)

weights_l = np.flipud(w_l[-1])
weights_r = np.flipud(w_r[-1])

fig = plt.figure(figsize=(9,5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

ax3 = plt.subplot(211)
plt.title('Final left weights')
plt.imshow(weights_l, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax3.text(i,j,int(label),ha='center',va='center', fontsize=8)


ax4 = plt.subplot(212)
plt.title('Final right weights')
plt.imshow(weights_r, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax4.text(i,j,int(label),ha='center',va='center', fontsize=8)


for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(20)

for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
             ax4.get_xticklabels() + ax4.get_yticklabels()):
    item.set_fontsize(20)

# fig.subplots_adjust(hspace=5.5)
# fig.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.42, right=0.98, left=0.11, bottom=0.04, top=0.9)
plt.savefig(path + '/plot_final_weights.pdf', bbox_inches="tight")
plt.show()
