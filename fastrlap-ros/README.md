# FastRLAP ROS
This directory contains a ROS implementation of FastRLAP, as well as the associated Gazebo environment.

## Instructions (simulator)
Make sure you have ROS 1 Noetic installed, including Gazebo. Also, make sure to install the `ros-noetic-jackal-*` packages.

In a clean `catkin_ws`, execute the following commands (note: we suggest doing this in a virtual Python environment using `virtualenv` or `mamba`/`conda`):

```bash
mkdir src
git clone git@github.com:kylestach/offroad-learning src/offroad_learning
git clone git@github.com:kylestach/cpr_gazebo src/cpr_gazebo --branch custom-maps
git clone git@github.com:ros-drivers/ackermann_msgs src/ackermann_msgs
git clone git@github.com:nilseuropa/realsense_ros_gazebo src/realsense-ros-gazebo

git clone git@github.com:kylestach/jaxrl5 jaxrl5 --branch offroad
cd jaxrl5
pip install -e . --no-deps
# Install JAX using pip, or conda/mamba install if preferred.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax gym absl-py tensorflow-probability ml_collections wandb tqdm git+https://github.com/qboticslabs/ros_numpy tensorflow
```

Next, download the [pretrained encoder](https://drive.google.com/file/d/11F9HU_H3cv20uAlm3L3wVNsAX7lcz374/view?usp=sharing) to `~/encoders/iql_encoder_checkpoint`.

Once you have run all of these commands, you should be able to run the simulator using the following commands:
```bash
export JACKAL_URDF_EXTRAS=/home/kyle/Documents/research/catkin_ws/src/offroad_learning/offroad_gazebo/urdf/jackal_custom.urdf.xacro
XLA_PYTHON_CLIENT_PREALLOCATE=false roslaunch offroad_learning sim_training_inference.launch
```

## Instructions (robot)
These instructions assume you have a car robot with the following components:
 - RedShift Labs UM7 inertial measurement unit (IMU)
 - UBlox ZED-F9P GPS
 - VESC speed controller
 - Jetson Xavier NX
 - Any V4L camera

First, install ROS 1 noetic. The following instructions should work out-of-the box: [http://wiki.ros.org/noetic/Installation/Ubuntu](http://wiki.ros.org/noetic/Installation/Ubuntu).

### Installing JAX
You will need to compile `jax` from scratch, because google does not provide the correct wheels. First, create a new virtual environment:
```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install numpy scipy six wheel
```

Clone the repository from `github.com/google/jax`. Then, with the virtual environment still sourced, run:
```bash
python build/build.py --cuda_compute_capabilities=sm_86
```
This step will take a long time. Once it is complete, run:
```bash
python3 -m pip install ./dist/*.whl
python3 -m pip install . -e
```

You can tell that `jax` is working with the following commands:
```bash
 $ python
Python 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.random.split(jax.random.PRNGKey(0))
>>> jax.zeros((1,)).get_device()
StreamExecutor(...)
```
If the above commands run without error, and the final one returns a `StreamExecutor, jax indeed works in GPU-accelerated mode on your Jetson!

### Setting up the workspace
In a new `catkin_ws`, run the following commands:

```bash
mkdir src
git clone git@github.com:kylestach/offroad-learning src/offroad_learning
git clone git@github.com:kylestach/racecar-offroad src/racecar --branch offroad

git clone git@github.com:kylestach/jaxrl5 jaxrl5 --branch offroad
# TODO: Install dependencies for jaxrl5
```
