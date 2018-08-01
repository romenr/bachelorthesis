# Overview

The network.py, environment.py and parameters.py contain the Network, Environment and Parameters.
training.py is used to train the controller. controller.py is used to run or evaluate the controller.
All plot_*.py files are used to generate plots.
training.sh and controller.sh are scripts that automatize training and evaluation.
Call a *.py file to show help

```bash
python training.py -h
```

# Setup

Give the training.sh and controller.sh file the rights to be executed.
```bash
sudo chmod +x training.sh controller.sh
```

# Run the experiment

The training.sh script will train and evaluate the controller. Then it will generate all plots.
The controller.sh script will take a trained controller and execute it.

Start roscore, v-rep and load the snake_follows_car_scene.ttt then start the simulation.

Run the experiment using the training script. The only parameter is the name of the directory
in witch all results will be saved e.g. ./data/session_001/

```bash
./training.sh session_001
```

All scripts can be executed manually by giving a directory as parameter.
