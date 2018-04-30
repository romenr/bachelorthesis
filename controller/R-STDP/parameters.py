#!/usr/bin/env python

# Resoltution of the input image for the SNN
view_resolution = [1, 1]

# Network simulation time resolution (in ms)
time_resolution = 0.1

# Connection Inisalisation min and max values
init_min_weight = 200
init_max_weight = 201

## STDP Dopamine Synapse Options

# Time Constant of reward signal
tau_n = 200.
# Time Constant of eligibility trace
tau_c = 1000.
# Minimum Weight Value
w_min = 0.
# Maximum Weight Value
W_max = 3000.
# Constant scaling strength of potentiation
A_plus = 1.
# Constant scaling strenght of depression
A_minus = 1.

