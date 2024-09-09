import itertools
import threading
import rospy
from message_filters import TimeSynchronizer, Subscriber
from functools import reduce
from sensor_msgs.msg import Image

class CustomApproximateTimeSynchronizer(TimeSynchronizer):
    def __init__(self, fs, queue_size, slop, allow_headerless=False):
        TimeSynchronizer.__init__(self, fs, queue_size)
        self.slop = rospy.Duration.from_sec(slop)
        self.allow_headerless = allow_headerless
        self.last_added = rospy.Time()

        # Set up a timer to periodically update parameters
        self.update_interval = rospy.Duration(1)  # Update every second
        self.timer = rospy.Timer(self.update_interval, self.update_parameters)

        # Initialize the command and topics
        self.command = 3  # Default command
        self.topics = self.determine_topics(self.command)
        self.update_topics(self.topics)

    def determine_topics(self, command):
        roi_name = [
            '/camera/front/image_raw',
            '/camera/front_right/image_raw',
            '/camera/front_left/image_raw',
            '/camera/back/image_raw',
            '/camera/back_left/image_raw',
            '/camera/back_right/image_raw'
        ]

        if command == 2:  # forward
            return [roi_name[0], roi_name[1], roi_name[2]]
        elif command == 1:  # turn left
            return [roi_name[0], roi_name[5], roi_name[2]]
        elif command == 0:  # turn right
            return [roi_name[0], roi_name[4], roi_name[1]]
        else:
            return roi_name

    def update_parameters(self, event):
        # Read the command and slop parameters from the ROS parameter server
        command_param = "/synchronizer/command"
        slop_param = "/synchronizer/slop"
        
        command = rospy.get_param(command_param, self.command)
        slop = rospy.get_param(slop_param, self.slop.to_sec())

        topics = self.determine_topics(command)
        # print(topics)

        # Update the synchronizer if parameters have changed
        if set(topics) != set(self.getTopics()) or slop != self.slop.to_sec():
            self.slop = rospy.Duration.from_sec(slop)
            self.update_topics(topics)

        self.command = command

    def update_topics(self, topics):
        # Update the input connections based on the new topics
        self.queues = [{} for _ in topics]
        self.input_connections = []
        for topic in topics:
            # print(topic)
            sub = Subscriber(topic, Image)
            sub.registerCallback(self.add, self.queues[len(self.input_connections)], len(self.input_connections))
            self.input_connections.append(sub)
            
    def getTopics(self):
        # print(self.input_connections)
        return [sub.resolved_name for sub in self.input_connections]

    def add(self, msg, my_queue, my_queue_index):
        if not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
            print(msg.header)
            if not self.allow_headerless:
                rospy.logwarn("Cannot use message filters with non-stamped messages. "
                              "Use the 'allow_headerless' constructor option to "
                              "auto-assign ROS time to headerless messages.")
                return
            stamp = rospy.Time.now()
        else:
            stamp = msg.header.stamp

        self.lock.acquire()
        my_queue[stamp] = msg

        # clear all buffers if jump backwards in time is detected
        now = rospy.Time.now()
        if now < self.last_added:
            rospy.loginfo("ApproximateTimeSynchronizer: Detected jump back in time. Clearing buffer.")
            for q in self.queues:
                q.clear()
        self.last_added = now

        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        
        # sort and leave only reasonable stamps for synchronization
        stamps = []
        for queue in self.queues:
            topic_stamps = []
            for s in queue:
                stamp_delta = abs(s - stamp)
                if stamp_delta > self.slop:
                    continue  # far over the slop
                topic_stamps.append((s, stamp_delta))
            if not topic_stamps:
                self.lock.release()
                return
            topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
            stamps.append(topic_stamps)

        for vv in itertools.product(*[next(iter(zip(*s))) for s in stamps]):
            vv = list(vv)
            # insert the new message
            if my_queue_index is not None:
                vv.insert(my_queue_index, stamp)
            qt = list(zip(self.queues, vv))
            if ((max(vv) - min(vv)) < self.slop) and (len([1 for q, t in qt if t not in q]) == 0):
                msgs = [q[t] for q, t in qt]
                self.signalMessage(*msgs)
                for q, t in qt:
                    del q[t]
                break  # fast finish after the synchronization
        self.lock.release()
