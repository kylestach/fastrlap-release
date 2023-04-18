from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['offroad_sensor_fusion'],
    package_dir={'': 'src'},
)

setup(**setup_args)