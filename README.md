# RT-BEV: Real-Time Bird's Eye View Perception for Autonomous Vehicles

## Overview

**RT-BEV** is an advanced system designed to provide real-time, vision-centric **Bird's Eye View (BEV)** perception for autonomous vehicles (AVs). By leveraging multiple cameras and deep learning models, RT-BEV constructs a 360-degree view of the environment, improving situational awareness, navigation, and decision-making for AVs.

### Key Features:
- **Real-time BEV perception** using a multi-camera setup.
- Enhanced situational awareness with 360-degree vision.
- Integration with ROS for seamless camera synchronization and data publishing.
- Pre-trained models for fast and efficient inference.

### End-to-End BEV Perception Pipeline

The following figure illustrates the end-to-end BEV perception pipeline used in RT-BEV:

![E2E BEV Perception Pipeline](./doc/figures/BEV-e2e-pipeline.pdf)

## System Design

The system is designed with a modular architecture, allowing efficient data processing, real-time synchronization, and seamless integration with autonomous vehicle systems. The core modules of RT-BEV include:

1. **Camera Synchronization**: A multi-camera synchronization node ensures that all camera feeds are aligned before processing.
2. **Inference Engine**: The BEV inference module processes the synchronized images to generate a bird's eye view.
3. **Data Publishing**: The camera images are published via ROS to allow real-time processing by other modules.

The design architecture is shown in the following figure:

![System Design](./doc/figures/RT-BEV-Design.pdf)

The system leverages state-of-the-art BEV perception algorithms to transform camera images into accurate overhead views, capturing critical details for navigation and obstacle avoidance.

## Implementation

The RT-BEV system is implemented using **Python** with **PyTorch** as the deep learning framework and **ROS** for real-time data processing and synchronization. Key components include:

- **Torch Inference**: PyTorch models are used to process the images and generate BEV outputs.
- **ROS Integration**: ROS nodes manage camera synchronization, image publishing, and BEV processing.

## Installation and Setup

To set up the environment, prepare the datasets, and run the system, please refer to the following guides:

- [Installation Guide](./doc/INSTALL.md)
- [Dataset Preparation Guide](./doc/DATA_PREP.md)
- [Running RT-BEV](./doc/RUN.md)

The **nuScenes V1.0 mini** dataset is already included in the Docker container, so no additional setup is required for testing.

### Installation Overview

Follow the steps in the [Installation Guide](./doc/INSTALL.md) to install all necessary dependencies. The guide will walk you through the environment setup using either a Docker container or a non-Docker approach.

A brief visualization of the installation and dataset preparation process:

![Installation and Dataset Preparation](./doc/figures/installation_dataset_prep.png)

## Usage Instructions

### For Docker Users:

1. **Step 1**: Open four terminals connected to the Docker container:
   ```bash
   docker exec -it container_name bash
   ```

2. **Step 2**: In Terminal 1, run the ROS master node:
   ```bash
   roscore
   ```

3. **Step 3**: In Terminal 2, run the RT-BEV inference node:
   ```bash
   cd UniAD
   ./tool/test_inference.sh
   ```

4. **Step 4**: In Terminal 3, run the multi-camera synchronization node:
   ```bash
   source /home/mobilitylab/catkin_ws/devel/setup.bash
   roslaunch rtbev_message_filters synchronizer.launch
   ```

5. **Step 5**: In Terminal 4, publish camera images:
   ```bash
   source /home/mobilitylab/catkin_ws/devel/setup.bash
   rosrun video_stream_opencv ros_publish_multi_cameras.py
   ```

For a more detailed explanation, refer to the [Running RT-BEV](./doc/RUN.md) guide.

## Results

RT-BEV has been evaluated on the **nuScenes V1.0** dataset, achieving real-time performance and high accuracy in generating BEV representations. Key results include:

- **Accurate BEV representations** with minimal latency.
- **Efficient multi-camera synchronization** ensuring smooth image processing.
- Robust handling of complex driving scenarios with multiple dynamic and static objects.

## Citation

If you use this work, please cite it using the following reference:

```
@article{RT-BEV2024,
  title={RT-BEV: Enhancing Real-Time BEV Perception for Autonomous Vehicles},
  author={Liangkai Liu and others},
  journal={IEEE Real-Time Systems Symposium (RTSS)},
  year={2024}
}
```

---

### Additional Documentation:
- [Installation Guide](./doc/INSTALL.md)
- [Dataset Preparation Guide](./doc/DATA_PREP.md)
- [Running RT-BEV](./doc/RUN.md)

---

### Figures and Illustrations

Ensure that the following images are placed in the `doc/figures/` directory:
1. `e2e_bev_pipeline.png`: End-to-end BEV perception pipeline.
2. `system_design.png`: RT-BEV system design.
3. `installation_dataset_prep.png`: Illustration of the installation and dataset preparation process.
