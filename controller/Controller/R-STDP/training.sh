#!/bin/bash
# Train the network, create plots and save all important data in the directory described by the first Parameter
dir="data/${1}"
data="${dir}/rstdp_data.h5"
mkdir ${dir}
python training.py -o ${data}
python plot_rstdp_weights.py -n -f ${data} -o "${dir}/weights.png"
python plot_rstdp_training.py -n -f ${data} -o "${dir}/training.png"
cp parameters.py ${dir}
