#!/bin/bash
# Uncomment this if you want verbose output. It prints every command before executing it.
set -x

# Train the network, create plots and save all important data in the directory described by the first Parameter
dir="data/${1}"
data="${dir}/training_data.h5"
eval="${dir}/evaluation_data.h5"

# Create the dir Directory if it doesn't exist
if [ ! -d "${dir}" ]; then
  echo "Creating directory ${dir}"
  mkdir ${dir}
fi

# Save the Parameters
echo "Saving parameters.py ..."
cp parameters.py ${dir}

# Train the Network
echo "Training Network, saving results in ${data}"
python training.py -o ${data}

# Create the Training plots
echo "Creating Training Plots"
python plot_rstdp_weights.py -n -f ${data} -o "${dir}/weights.png"
python plot_rstdp_training.py -n -f ${data} -o "${dir}/training.png"
python plot_rstdp_rewards.py -n -f ${data} -o "${dir}/rewards.png"

# Evaluate the Network
echo "Checking Network performance, saving results in ${eval}"
python controller.py -f ${data} -o ${eval}

# Create the Evaluation plots
echo "Creating Evaluation plots"
python plot_rstdp_eval.py -n -f ${eval} -o "${dir}/eval.png"
