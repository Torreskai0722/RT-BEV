# RT-BEV: Real-Time Bird's Eye View Perception for Autonomous Vehicles

## Overview

**RT-BEV** is an advanced system designed to provide real-time, vision-centric **Bird's Eye View (BEV)** perception for autonomous vehicles (AVs). By leveraging multiple cameras and deep learning models, RT-BEV constructs a 360-degree view of the environment, improving situational awareness, navigation, and decision-making for AVs.

### Key Features:
- **Real-time BEV perception** using a multi-camera setup.
- Enhanced situational awareness with 360-degree vision.
- Integration with ROS for seamless camera synchronization and data publishing.
- Pre-trained models for fast and efficient inference.

## System Design

The system is designed with a multi-module approach that enables efficient data processing, real-time synchronization, and seamless integration with autonomous vehicle systems. The core modules of the RT-BEV system include:

1. **Camera Synchronization**: A multi-camera synchronization node ensures that all camera feeds are aligned before processing.
2. **Inference Engine**: The BEV inference module processes the synchronized images to generate a bird's eye view.
3. **Data Publishing**: The camera images are published via ROS to allow real-time processing by other modules.

The system leverages state-of-the-art BEV perception algorithms, such as BEVFormer and others, to transform camera images into an accurate overhead view, capturing critical details for navigation and obstacle avoidance.

## Implementation

The RT-BEV system is implemented in Python, using **PyTorch** as the deep learning framework and **ROS** for real-time data processing and synchronization. Key implementation details include:

- **Torch Inference**: PyTorch-based models are used to process the images and generate BEV outputs.
- **ROS Integration**: ROS nodes manage camera synchronization, image publishing, and BEV processing, ensuring smooth communication between the different components.

The Docker image for RT-BEV already includes all necessary dependencies and pre-trained models, simplifying the setup process.

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

The **nuScenes V1.0 mini** dataset is pre-installed in the Docker container for testing purposes, and you can use it directly without additional setup.

### nuScenes V1.0 Full Dataset Setup (Optional):

If you wish to use the **nuScenes V1.0 full** dataset, follow these steps:

1. Download the dataset from [nuScenes](https://www.nuscenes.org/download) and place it in the `RT-BEV/nuscenes-full` directory.
2. Generate or download the necessary data information files:
   ```bash
   cd RT-BEV/nuscenes-full
   ./tools/uniad_create_data.sh
   ```
3. Place the pre-trained weights and motion anchors in the appropriate directories.

## Results

RT-BEV has been evaluated on the **nuScenes V1.0** dataset, achieving real-time performance and high accuracy in generating BEV representations. Key results include:

- **Accurate BEV representations** with minimal latency.
- **Efficient multi-camera synchronization** ensuring smooth image processing.
- Robust handling of complex driving scenarios with multiple dynamic and static objects.

Detailed evaluation results can be found in our publication.

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
