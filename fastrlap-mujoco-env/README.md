# Offroad Simulator for Mujoco
This is a simulator for offroad driving. It is built with the Mujoco simulation engine and the OpenAI Gym interface. It is designed to be used with the Jaxrl implementation of RL algorithms.

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGM3ODhmY2M2ZDQwM2RhNTlmZDAxN2I0ZDZkNGUzZDIwYzcwMDRjOSZjdD1n/T2NKLcbSpgvWBbhr71/giphy.gif)


## Installation
To install Jaxrl and the required dependencies for the simulator, run the included installation bash script:
```
chmod +x install.sh
bash install.sh
```
This installation will install the dependecies for Jaxrl, the mujoco simulator as well as initialize the heightmap data for the simulators terrain.

## Usage
First, we need to collect some training data. To do this, we can run the following command:
```
python data_collect.py --dataset_file <path_to_dataset_file>
```
The controls for the car are as follows:
- ↑: Accelerate forwards
- ↓: Accelerate backwards
- ←: Steer left
- →: Steer right

Currently, our goal graph definitions are location in the `goal_graph.py` file. To change the goal graph, simply change the `goal_config` variable in the `task.py` file.

This will save a memory buffer to your desired location with the collected data. This data can then be used to train a model using the included training script:
To run the example training script, run
```
XLA_PYTHON_CLIENT_PREALLOCATE=false train_online_pixels.py # for training on pixels
XLA_PYTHON_CLIENT_PREALLOCATE=false train_online_states.py # for training on state
```

## Acknowledgements
This environment relies on assets derived from [MuSHR](mushr.io) and the (free) [Network of Paths](https://www.turbosquid.com/3d-models/3d-paths-blender-1708806) model.
