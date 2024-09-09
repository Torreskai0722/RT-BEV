;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::DetectObject)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'DetectObject (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::DETECTOBJECT")
  (make-package "OBJECT_MSGS::DETECTOBJECT"))
(unless (find-package "OBJECT_MSGS::DETECTOBJECTREQUEST")
  (make-package "OBJECT_MSGS::DETECTOBJECTREQUEST"))
(unless (find-package "OBJECT_MSGS::DETECTOBJECTRESPONSE")
  (make-package "OBJECT_MSGS::DETECTOBJECTRESPONSE"))

(in-package "ROS")





(defclass object_msgs::DetectObjectRequest
  :super ros::object
  :slots (_image_paths ))

(defmethod object_msgs::DetectObjectRequest
  (:init
   (&key
    ((:image_paths __image_paths) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _image_paths __image_paths)
   self)
  (:image_paths
   (&optional __image_paths)
   (if __image_paths (setq _image_paths __image_paths)) _image_paths)
  (:serialization-length
   ()
   (+
    ;; string[] _image_paths
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _image_paths)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _image_paths
     (write-long (length _image_paths) s)
     (dolist (elem _image_paths)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _image_paths
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _image_paths (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _image_paths i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(defclass object_msgs::DetectObjectResponse
  :super ros::object
  :slots (_objects ))

(defmethod object_msgs::DetectObjectResponse
  (:init
   (&key
    ((:objects __objects) (let (r) (dotimes (i 0) (push (instance object_msgs::ObjectsInBoxes :init) r)) r))
    )
   (send-super :init)
   (setq _objects __objects)
   self)
  (:objects
   (&rest __objects)
   (if (keywordp (car __objects))
       (send* _objects __objects)
     (progn
       (if __objects (setq _objects (car __objects)))
       _objects)))
  (:serialization-length
   ()
   (+
    ;; object_msgs/ObjectsInBoxes[] _objects
    (apply #'+ (send-all _objects :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; object_msgs/ObjectsInBoxes[] _objects
     (write-long (length _objects) s)
     (dolist (elem _objects)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; object_msgs/ObjectsInBoxes[] _objects
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objects (let (r) (dotimes (i n) (push (instance object_msgs::ObjectsInBoxes :init) r)) r))
     (dolist (elem- _objects)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass object_msgs::DetectObject
  :super ros::object
  :slots ())

(setf (get object_msgs::DetectObject :md5sum-) "11b2ac83822624e9953f8718798e5220")
(setf (get object_msgs::DetectObject :datatype-) "object_msgs/DetectObject")
(setf (get object_msgs::DetectObject :request) object_msgs::DetectObjectRequest)
(setf (get object_msgs::DetectObject :response) object_msgs::DetectObjectResponse)

(defmethod object_msgs::DetectObjectRequest
  (:response () (instance object_msgs::DetectObjectResponse :init)))

(setf (get object_msgs::DetectObjectRequest :md5sum-) "11b2ac83822624e9953f8718798e5220")
(setf (get object_msgs::DetectObjectRequest :datatype-) "object_msgs/DetectObjectRequest")
(setf (get object_msgs::DetectObjectRequest :definition-)
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

string[] image_paths      # image path list for inference
---
ObjectsInBoxes[] objects  # bounding box list for each of inferred images


================================================================================
MSG: object_msgs/ObjectsInBoxes
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

(setf (get object_msgs::DetectObjectResponse :md5sum-) "11b2ac83822624e9953f8718798e5220")
(setf (get object_msgs::DetectObjectResponse :datatype-) "object_msgs/DetectObjectResponse")
(setf (get object_msgs::DetectObjectResponse :definition-)
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

string[] image_paths      # image path list for inference
---
ObjectsInBoxes[] objects  # bounding box list for each of inferred images


================================================================================
MSG: object_msgs/ObjectsInBoxes
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



(provide :object_msgs/DetectObject "11b2ac83822624e9953f8718798e5220")


