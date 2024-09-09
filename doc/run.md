# RT-BEV: How to Run

This guide explains how to run the **RT-BEV** system for both Docker and non-Docker users. For Docker users, you'll need to open four terminals connected to the same Docker container and run the corresponding commands in each terminal.

---

### For Docker Users:

Launch RT-BEV docker image:

```bash
./docker/rt-bev.sh
```

#### Step 1: Open Four Terminals

To begin, open **four** separate terminals and connect them to the same running Docker container by using the following command in each terminal:

```bash
docker exec -it container_name bash
```

Replace `container_name` with the name or ID of your running container. Once connected, follow the instructions below for each terminal:

#### Terminal 1: Run ROS Master Node

In the first terminal, start the **ROS master node** by running:

```bash
roscore
```

This initializes the ROS environment and starts the communication system required for nodes to interact.

#### Terminal 2: Run RT-BEV Inference Node

In the second terminal, navigate to the UniAD directory and start the **RT-BEV inference node** by executing:

```bash
cd UniAD
./tool/test_inference.sh
```

This command runs the RT-BEV inference using pre-trained models, processing the input camera data.

#### Terminal 3: Run Multi-Camera Synchronization Node

In the third terminal, initialize the **multi-camera synchronization node**. First, source your ROS workspace and then launch the node:

```bash
source /home/mobilitylab/catkin_ws/devel/setup.bash
roslaunch rtbev_message_filters synchronizer.launch
```

This node ensures that multiple camera streams are synchronized before processing.

#### Terminal 4: Run Camera Images Publishing Node

In the fourth terminal, source the ROS workspace and then run the camera image publishing script to send multi-camera video streams to ROS:

```bash
source /home/mobilitylab/catkin_ws/devel/setup.bash
rosrun video_stream_opencv ros_publish_multi_cameras.py
```

This node publishes the camera images, making them available for RT-BEV to process.

---

### For Non-Docker Users:

If you are not using Docker, make sure to follow the **installation steps** outlined in the [Installation Guide](./INSTALLATION.md). After the environment is properly set up, follow the same process as described above but without the need to open Docker terminals.

1. **Terminal 1**: Run `roscore` to start the ROS master.
2. **Terminal 2**: Navigate to the UniAD directory and run `./tool/test_inference.sh` to start the inference.
3. **Terminal 3**: Source the workspace and run `roslaunch rtbev_message_filters synchronizer.launch` for multi-camera synchronization.
4. **Terminal 4**: Source the workspace and run `rosrun video_stream_opencv ros_publish_multi_cameras.py` to publish the camera images.

---

### Notes:

- Ensure that all the ROS nodes are communicating correctly by verifying that no errors appear in any of the terminals.
- The system processes camera images in real-time, so make sure the camera streams are active and correctly configured before running the inference.
