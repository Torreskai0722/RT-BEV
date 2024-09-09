;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::ObjectsInBoxes)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'ObjectsInBoxes (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::OBJECTSINBOXES")
  (make-package "OBJECT_MSGS::OBJECTSINBOXES"))

(in-package "ROS")
;;//! \htmlinclude ObjectsInBoxes.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass object_msgs::ObjectsInBoxes
  :super ros::object
  :slots (_header _objects_vector _inference_time_ms ))

(defmethod object_msgs::ObjectsInBoxes
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:objects_vector __objects_vector) (let (r) (dotimes (i 0) (push (instance object_msgs::ObjectInBox :init) r)) r))
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
    ;; object_msgs/ObjectInBox[] _objects_vector
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
     ;; object_msgs/ObjectInBox[] _objects_vector
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
   ;; object_msgs/ObjectInBox[] _objects_vector
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objects_vector (let (r) (dotimes (i n) (push (instance object_msgs::ObjectInBox :init) r)) r))
     (dolist (elem- _objects_vector)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float32 _inference_time_ms
     (setq _inference_time_ms (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get object_msgs::ObjectsInBoxes :md5sum-) "2c070e019267e39d554a36de7d183780")
(setf (get object_msgs::ObjectsInBoxes :datatype-) "object_msgs/ObjectsInBoxes")
(setf (get object_msgs::ObjectsInBoxes :definition-)
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

# This message can represent objects detected and their bounding-boxes in a frame
std_msgs/Header header        # timestamp in header is the time the sensor captured the raw data
ObjectInBox[] objects_vector  # ObjectInBox array
float32 inference_time_ms     # inference time of this frame. the unit is millisecond.

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
MSG: object_msgs/ObjectInBox
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

# This message can represent a detected object and its region of interest
Object object                     # detected object
sensor_msgs/RegionOfInterest roi  # region of interest

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

================================================================================
MSG: sensor_msgs/RegionOfInterest
# This message is used to specify a region of interest within an image.
#
# When used to specify the ROI setting of the camera when the image was
# taken, the height and width fields should either match the height and
# width fields for the associated image; or height = width = 0
# indicates that the full resolution image was captured.

uint32 x_offset  # Leftmost pixel of the ROI
                 # (0 if the ROI includes the left edge of the image)
uint32 y_offset  # Topmost pixel of the ROI
                 # (0 if the ROI includes the top edge of the image)
uint32 height    # Height of ROI
uint32 width     # Width of ROI

# True if a distinct rectified ROI should be calculated from the \"raw\"
# ROI in this message. Typically this should be False if the full image
# is captured (ROI not used), and True if a subwindow is captured (ROI
# used).
bool do_rectify

")



(provide :object_msgs/ObjectsInBoxes "2c070e019267e39d554a36de7d183780")


