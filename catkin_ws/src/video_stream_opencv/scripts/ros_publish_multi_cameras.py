#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import os
import signal
import sys

def quit(signum, frame):
    print('')
    print('stop publishing images')
    sys.exit()

def extract_timestamps(base_directory):
    timestamps = {}
    sensor_folders = [
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"
    ]
    
    for sensor in sensor_folders:
        folder_path = os.path.join(base_directory, sensor)
        timestamps[sensor] = []
        if not os.path.exists(folder_path):
            rospy.logwarn("Folder %s does not exist." % folder_path)
            continue
        
        sorted_filenames = sorted(os.listdir(folder_path))
        for filename in sorted_filenames:
            parts = filename.split("__")
            if len(parts) == 3:
                timestamp = float(parts[2].split('.')[0]) / 1e6
                timestamps[sensor].append((timestamp, filename))
            else:
                rospy.loginfo("Unexpected filename format: %s" % filename)
    
    return timestamps

def publish_images(image_paths, timestamps, publisher, bridge, camera_name):
    for t, filename in sorted(timestamps):
        img_path = os.path.join(image_paths, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
            # img_msg.header.stamp = rospy.Time.from_sec(t)
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = camera_name
            publisher.publish(img_msg)
            rospy.sleep(0.1)

def camera_thread(image_paths, timestamps, publisher, bridge, camera_name):
    while not rospy.is_shutdown():
        publish_images(image_paths, timestamps, publisher, bridge, camera_name)
        rospy.loginfo(f"Finished publishing images for {camera_name}.")
        break  # Exit the loop after publishing all images

def main():
    rospy.init_node('multi_camera_publisher', anonymous=True)
    bridge = CvBridge()

    base_directory = "/home/mobilitylab/v1.0-mini/sweeps"
    timestamps = extract_timestamps(base_directory)

    publishers = {
        'CAM_BACK': rospy.Publisher('/camera/back/image_raw', Image, queue_size=10),
        'CAM_BACK_LEFT': rospy.Publisher('/camera/back_left/image_raw', Image, queue_size=10),
        'CAM_BACK_RIGHT': rospy.Publisher('/camera/back_right/image_raw', Image, queue_size=10),
        'CAM_FRONT': rospy.Publisher('/camera/front/image_raw', Image, queue_size=10),
        'CAM_FRONT_LEFT': rospy.Publisher('/camera/front_left/image_raw', Image, queue_size=10),
        'CAM_FRONT_RIGHT': rospy.Publisher('/camera/front_right/image_raw', Image, queue_size=10)
    }
    
    paths = {
        'CAM_BACK': os.path.join(base_directory, "CAM_BACK"),
        'CAM_BACK_LEFT': os.path.join(base_directory, "CAM_BACK_LEFT"),
        'CAM_BACK_RIGHT': os.path.join(base_directory, "CAM_BACK_RIGHT"),
        'CAM_FRONT': os.path.join(base_directory, "CAM_FRONT"),
        'CAM_FRONT_LEFT': os.path.join(base_directory, "CAM_FRONT_LEFT"),
        'CAM_FRONT_RIGHT': os.path.join(base_directory, "CAM_FRONT_RIGHT")
    }

    threads = []
    for key in publishers.keys():
        thread = threading.Thread(target=camera_thread, args=(paths[key], timestamps[key], publishers[key], bridge, key))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    
    main()
