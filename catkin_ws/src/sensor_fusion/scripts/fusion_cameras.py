#! /usr/bin/env python3.6
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import signal
import sys

def quit(signum, frame):
    print('')
    print('stop cameras fusion')
    sys.exit()

def image_callback(back, back_left, back_right, front, front_left, front_right):
    # This callback will be called when synchronized messages are available
    bridge = CvBridge()

    # Convert ROS images to OpenCV format
    back_image = bridge.imgmsg_to_cv2(back, desired_encoding='bgr8')
    back_left_image = bridge.imgmsg_to_cv2(back_left, desired_encoding='bgr8')
    back_right_image = bridge.imgmsg_to_cv2(back_right, desired_encoding='bgr8')
    front_image = bridge.imgmsg_to_cv2(front, desired_encoding='bgr8')
    front_left_image = bridge.imgmsg_to_cv2(front_left, desired_encoding='bgr8')
    front_right_image = bridge.imgmsg_to_cv2(front_right, desired_encoding='bgr8')

    # Extract timestamps from each image message header
    timestamps = [
        back.header.stamp.to_sec(),
        back_left.header.stamp.to_sec(),
        back_right.header.stamp.to_sec(),
        front.header.stamp.to_sec(),
        front_left.header.stamp.to_sec(),
        front_right.header.stamp.to_sec()
    ]

    # Calculate min and max timestamps
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    time_span = max_timestamp - min_timestamp

    # Open file and append the fusion results in CSV format
    with open("fusion_results_ros_comm.csv", "a+") as f:
        f.write(f"{min_timestamp},{max_timestamp},{time_span}\n")

    # Placeholder for additional processing
    print("Logged fusion data to file.")
    
def main():
    rospy.init_node('image_fusion_node', anonymous=True)

    # Create subscribers for each camera image topic
    back_sub = message_filters.Subscriber('/camera/back/image_raw', Image)
    back_left_sub = message_filters.Subscriber('/camera/back_left/image_raw', Image)
    back_right_sub = message_filters.Subscriber('/camera/back_right/image_raw', Image)
    front_sub = message_filters.Subscriber('/camera/front/image_raw', Image)
    front_left_sub = message_filters.Subscriber('/camera/front_left/image_raw', Image)
    front_right_sub = message_filters.Subscriber('/camera/front_right/image_raw', Image)

    # Synchronize the subscribers with a time synchronizer
    sync = message_filters.ApproximateTimeSynchronizer([back_sub, back_left_sub, back_right_sub, front_sub, front_left_sub, front_right_sub], queue_size=10, slop=0.2)
    sync.registerCallback(image_callback)

    rospy.loginfo("Fusion node has started, subscribing to image topics...")
    rospy.spin()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    main()

