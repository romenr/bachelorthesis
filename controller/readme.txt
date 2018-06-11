The V-REP_Project_R-STDP folder contains all customized files for running the scenario file. In order to make it run correctly, it is necessary to set up the v-rep environment such that it is able to connect to ROS.

################################

Run Experiment:
	1)	Launch ROS by calling "roscore"
	2)	Launch V-REP and ensure that it succesfully loads the ROSInterface module
	3)	Load the snake_follows_car_scene.ttt file
	4)	Run either controller.py or training.py from the project folder (it is necessary to have a data path to load and save the network parameters)
	5)	Start the Scene in V-REP

################################

The python code files which implement the R-STDP approach are adapted from the Master's Thesis. The original files can be found in the extra folder.

The signal flow between the snake and the spiking neural network is the following:
	1) Snake -> SNN:	The camera image from the snake head is filtered for red pixels and then downsampled to a feeding size of 8*4 pixels. The number of red pixels in one field of the 8*4 array is then used as an excitation rate for the corresponding input neuron.
	2) SNN -> Snake:	The excitation of the two output neurons are interpreted as "left wheel" and "right wheel" velocity respectively. After low-pass filtering this signal, it is transformed into a turning radius for the snake.

For training purpose, the "supervisor" has to compute a reward for the current state which is computed out of the camera image. In the current setup, the reward value is linearly proportional to the horizontal centroid position of the red pixels in the image. The reward is 1 if the centroid is in the center and 0 if it is at the boundary. The reset criterion is fulfilled if there is no more red pixels in the image.

As an additional control loop, the snake keeps a certain distance to the red car by using a sonar range sensor.

################################ 

In the current setup, the network weights do not seem to converge.

Open work:
	1)	Analysis of the failing experiment and appropriate adaptions
	2)	Defining two or more trajectories for the red car in order to have different tracks for training
