#!/bin/bash

# Install Jaxrl5 and corresponding dependencies
cd jaxrl5
pip install -r requirements.txt
pip install git+https://github.com/ikostrikov/dmcgym.git --no-dependencies
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .

# Dependencies of offroad_sim_v2
cd ..
pip install -r requirements.txt

# Initialize the heightmap data
cd offroad_sim_v2
python convert_heightmap.py
cd ..