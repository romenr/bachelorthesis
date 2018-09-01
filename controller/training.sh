#!/bin/bash
# Uncomment this if you want verbose output. It prints every command before executing it.
set -x

# Train the network, create plots and save all important data in the directory described by the first Parameter
dir="data/${1}"

# Create the dir Directory if it doesn't exist
if [ ! -d "${dir}" ]; then
  echo "Creating directory ${dir}"
  mkdir ${dir}
fi

# Save the Parameters
echo "Saving parameters.py ..."
cp parameters.py ${dir}

# Train the Network
echo "Training Network, saving results in ${dir}"
python training.py tf ${dir}

# Create the Training plots
echo "Creating Training Plots"
python plot_rstdp_training.py tf -n ${dir}

# Evaluate the Network
echo "Checking Network performance, saving results in ${dir}"
python controller.py tf ${dir}

# Create the Evaluation plots
echo "Creating Evaluation plots"
python plot_rstdp_eval.py -n ${dir}
