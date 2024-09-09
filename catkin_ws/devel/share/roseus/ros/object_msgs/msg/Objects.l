;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::Objects)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'Objects (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::OBJECTS")
  (make-package "OBJECT_MSGS::OBJECTS"))

(in-package "ROS")
;;//! \htmlinclude Objects.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass object_msgs::Objects
  :super ros::object
  :slots (_header _objects_vector _inference_time_ms ))

(defmethod object_msgs::Objects
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:objects_vector __objects_vector) (let (r) (dotimes (i 0) (push (instance object_msgs::Object :init) r)) r))
    ((:inference_time_ms __inference_time_ms) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _objects_vector __objects_vector)
   (setq _inference_time_ms (float __inference_time_ms))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:objects_vector
   (&rest __objects_vector)
   (if (keywordp (car __objects_vector))
       (send* _objects_vector __objects_vector)
     (progn
       (if __objects_vector (setq _objects_vector (car __objects_vector)))
       _objects_vector)))
  (:inference_time_ms
   (&optional __inference_time_ms)
   (if __inference_time_ms (setq _inference_time_ms __inference_time_ms)) _inference_time_ms)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; object_msgs/Object[] _objects_vector
    (apply #'+ (send-all _objects_vector :serialization-length)) 4
    ;; float32 _inference_time_ms
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; object_msgs/Object[] _objects_vector
     (write-long (length _objects_vector) s)
     (dolist (elem _objects_vector)
       (send elem :serialize s)
       )
     ;; float32 _inference_time_ms
       (sys::poke _inference_time_ms (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; object_msgs/Object[] _objects_vector
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objects_vector (let (r) (dotimes (i n) (push (instance object_msgs::Object :init) r)) r))
     (dolist (elem- _objects_vector)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float32 _inference_time_ms
     (setq _inference_time_ms (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get object_msgs::Objects :md5sum-) "491e959c05e1aac76496e491368f3ef1")
(setf (get object_msgs::Objects :datatype-) "object_msgs/Objects")
(setf (get object_msgs::Objects :definition-)
      "# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This message can represent objects detected in a frame
std_msgs/Header header    # timestamp in header is the time the sensor captured the raw data
Object[] objects_vector   # Object array
float32 inference_time_ms # inference time of this frame. the unit is millisecond.

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: object_msgs/Object
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This message define the property of detected object
string object_name  # object name
float32 probability # probability of detected object

")



(provide :object_msgs/Objects "491e959c05e1aac76496e491368f3ef1")


