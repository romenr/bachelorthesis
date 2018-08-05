#!/bin/bash
# Uncomment this if you want verbose output. It prints every command before executing it.
set -x

# Execute the network from dir
dir="data/${1}"

# Load the parameters
echo "Loading parameters.py ..."
cp "${dir}/parameters.py" .

# Execute the controller
echo "Executing controller, saving results in ${dir}"
python controller.py ${dir}

# Create the Evaluation plots
echo "Creating Evaluation plots"
python plot_rstdp_eval.py -n ${dir}
