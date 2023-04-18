# FastRLAP Code Release
This repository contains an implementation of FastRLAP for ROS robots, with associated gazebo environments, as well as a MuJoCo environment for offroad driving. To cite the FastRLAP paper, please use the following reference:
```bibtex
@article{stachowicz2023fastrlap,
  title={FastRLAP: A System for Learning High-Speed Driving via Deep RL and Autonomous Practicing},
  author={Stachowicz, Kyle and Bhorkar, Arjun and Shah, Dhruv and Kostrikov, Ilya and Levine, Sergey},
  journal={arXiv preprint arXiv:TODO},
  year={2023}
}
```

The ROS implementation, along with installation instructions is in the `fastrlap-ros` directory. The MuJoCo environment is located in the `fastrlap-mujoco-env` directory. The shared implementations of (several) RL algorithms is included in the `jaxrl5` directory.
