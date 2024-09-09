#!/root/anaconda3/envs/uniad/bin/python3.8

import rospy
from rtbev_message_filters.custom_approx_time_sync import CustomApproximateTimeSynchronizer
from message_filters import Subscriber
from sensor_msgs.msg import Image
from std_msgs.msg import String

def callback(pub, *msgs):
    # Process synchronized messages here
    # print("Received synchronized messages")
    timestamps = []
    seqs = []
    frame_ids = []

    for msg in msgs:
        timestamps.append(msg.header.stamp.to_sec())
        frame_ids.append(msg.header.frame_id)
        # print(msg.header.frame_id)
        # print(msg.header.seq)
        seqs.append(msg.header.seq)
    
    tmin = min(timestamps)
    tnow = rospy.Time.now().to_sec()
    command = rospy.get_param("/synchronizer/command")
    print(tnow - tmin, tnow, seqs, command)
    seq_min = min(seqs)
    rospy.set_param('/synchronization/frame_id', seq_min)

    # Publish the synchronized frame_id
    # pub.publish(str(seq_min))

if __name__ == '__main__':
    rospy.init_node('synchronizer_node')

    # Publisher to publish the synchronized frame_id
    frame_id_pub = rospy.Publisher('/synchronization/frame_id', String, queue_size=10)

    # Get the initial command and slop from the parameter server
    command = rospy.get_param("/synchronizer/command", 3)
    slop = rospy.get_param("/synchronizer/slop", 0.1)

    # Determine initial topics based on the command
    roi_name = [
        '/camera/front/image_raw',
        '/camera/front_right/image_raw',
        '/camera/front_left/image_raw',
        '/camera/back/image_raw',
        '/camera/back_left/image_raw',
        '/camera/back_right/image_raw'
    ]
    if command == 2:
        topics = [roi_name[0], roi_name[1], roi_name[2]]
    elif command == 1:
        topics = [roi_name[0], roi_name[5], roi_name[2]]
    elif command == 0:
        topics = [roi_name[0], roi_name[4], roi_name[1]]
    else:
        topics = roi_name

    # Create subscribers for initial topics
    subscribers = [Subscriber(topic, Image) for topic in topics]
    ats = CustomApproximateTimeSynchronizer(subscribers, queue_size=10, slop=slop)

     # Register callback using lambda to include the publisher
    ats.registerCallback(lambda *msgs: callback(frame_id_pub, *msgs))
    rospy.spin()
