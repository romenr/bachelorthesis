#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import path
import argparse
import parameters as param
import math

data_x = np.linspace(-math.degrees(0.6), math.degrees(0.6), 100, dtype=float)
data_y = np.array([np.linspace(-0.6, 0.6, 100, dtype=float), np.linspace(0.6, -0.6, 100, dtype=float)]).swapaxes(0, 1)

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)

ax1 = plt.subplot(gs[0, 0])

plt.grid(True)
ax1.plot(data_x, data_y)
ax1.set_xlim((-math.degrees(0.6), math.degrees(0.6)))
ax1.set_ylim((-0.6, 0.6))
ax1.set_xlabel(u"α")
ax1.set_ylabel("Dopamine Reward")
ax1.set_xticks([-30, -20, -10, 0, 10, 20, 30])
ax1.set_xticklabels([u'-30°', u'-20°', u'-10°', u'0°', u'10°', u'20°', u'30°'])
ax1.legend(["Left Neuron", "Right Neuron"])

fig.tight_layout()
plt.savefig(path.join("data", "angle_reward.pdf"))
plt.show()
