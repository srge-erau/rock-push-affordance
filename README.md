# Learning Rock Pushability on Rough Planetary Terrain

This repository contains the ROS2 implementation for the research paper "Learning Rock Pushability on Rough Planetary Terrain," presented at the 2025 IEEE ICRA Workshop on Field Robotics. Please refer to the paper for further detail.

---

## Core Concepts

This work moves beyond traditional avoidance by actively manipulating the environment. Key concepts include:

* **Affordance Learning:** The robot learns the "affordance" of an obstacle—what actions it offers. Specifically, can it be pushed?
* **Exteroceptive & Proprioceptive Feedback:** The system combines visual cues (size, shape, terrain slope) with physical force feedback from pushing the object to make a decision.
* **Probabilistic Framework:** The decision of whether an obstacle can be relocated is based on a probabilistic model, accounting for the uncertainties of unstructured environments.

---

## Repository Structure

This repository is organized as a collection of ROS2 Humble packages within a `src` directory.

* `boeing_interfaces`: Contains custom ROS2 message (`.msg`) and service (`.srv`) definitions used for communication between the different nodes in the system.
* `boeing_vision`: This package is responsible for the visual perception pipeline. It likely handles obstacle detection, localization, and analysis of the surrounding terrain from camera data.
* **[Coming Soon]**: A third package for Gazebo simulation will be added.

---

## Getting Started

### Prerequisites

* Ubuntu 22.04
* [ROS2 Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html)
* `colcon` build tools
* `rosdep`

### Dependencies

### System Dependencies
* **Ubuntu 22.04**
* **ROS2 Humble Hawksbill**: The core robotics middleware.
* **Intel RealSense SDK 2.0**: Must be installed before building the ROS wrapper. Follow the official installation guide.
* **Intel RealSense ROS2 Wrapper**: The ROS2 driver for RealSense cameras.
* **ros-humble-cv-bridge**: ROS package to convert between ROS Image messages and OpenCV images.

### Python Dependencies
* **numpy**
* **scipy**
* **matplotlib**
* **open3d**
* **opencv-python**

## Usage

To run the full system, you can use the provided launch files.

```bash
ros2 run boeing_vision normal_estimator.py
ros2 run boeing_vision filter_pointcloud.py
ros2 run boeing_vision pointcloud_segmentation.py
ros2 run boeing_vision obstacle_feature_extractor.py
```
---
If you use this work in your research, please cite the original paper:

```
@inproceedings{Girgin2025ICRA,
  author    = {Girgin, Tuba and Girgin, Emre and Kilic, Cagri},
  title     = {Learning Rock Pushability on Rough Planetary Terrain},
  booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA) Workshop on Field Robotics},
  year      = {2025},
  address   = {Daytona Beach, FL, USA},
  publisher = {IEEE}
}
```

