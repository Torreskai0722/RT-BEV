;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::ClassifyObject)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'ClassifyObject (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::CLASSIFYOBJECT")
  (make-package "OBJECT_MSGS::CLASSIFYOBJECT"))
(unless (find-package "OBJECT_MSGS::CLASSIFYOBJECTREQUEST")
  (make-package "OBJECT_MSGS::CLASSIFYOBJECTREQUEST"))
(unless (find-package "OBJECT_MSGS::CLASSIFYOBJECTRESPONSE")
  (make-package "OBJECT_MSGS::CLASSIFYOBJECTRESPONSE"))

(in-package "ROS")





(defclass object_msgs::ClassifyObjectRequest
  :super ros::object
  :slots (_image_paths ))

(defmethod object_msgs::ClassifyObjectRequest
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

(defclass object_msgs::ClassifyObjectResponse
  :super ros::object
  :slots (_objects ))

(defmethod object_msgs::ClassifyObjectResponse
  (:init
   (&key
    ((:objects __objects) (let (r) (dotimes (i 0) (push (instance object_msgs::Objects :init) r)) r))
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
    ;; object_msgs/Objects[] _objects
    (apply #'+ (send-all _objects :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; object_msgs/Objects[] _objects
     (write-long (length _objects) s)
     (dolist (elem _objects)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; object_msgs/Objects[] _objects
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objects (let (r) (dotimes (i n) (push (instance object_msgs::Objects :init) r)) r))
     (dolist (elem- _objects)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass object_msgs::ClassifyObject
  :super ros::object
  :slots ())

(setf (get object_msgs::ClassifyObject :md5sum-) "530ab171b0ce2b4875431fb220cd7b7f")
(setf (get object_msgs::ClassifyObject :datatype-) "object_msgs/ClassifyObject")
(setf (get object_msgs::ClassifyObject :request) object_msgs::ClassifyObjectRequest)
(setf (get object_msgs::ClassifyObject :response) object_msgs::ClassifyObjectResponse)

(defmethod object_msgs::ClassifyObjectRequest
  (:response () (instance object_msgs::ClassifyObjectResponse :init)))

(setf (get object_msgs::ClassifyObjectRequest :md5sum-) "530ab171b0ce2b4875431fb220cd7b7f")
(setf (get object_msgs::ClassifyObjectRequest :datatype-) "object_msgs/ClassifyObjectRequest")
(setf (get object_msgs::ClassifyObjectRequest :definition-)
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

string[] image_paths  # image path list for inference
---
Objects[] objects     # bounding box list for each of inferred images


================================================================================
MSG: object_msgs/Objects
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

(setf (get object_msgs::ClassifyObjectResponse :md5sum-) "530ab171b0ce2b4875431fb220cd7b7f")
(setf (get object_msgs::ClassifyObjectResponse :datatype-) "object_msgs/ClassifyObjectResponse")
(setf (get object_msgs::ClassifyObjectResponse :definition-)
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

string[] image_paths  # image path list for inference
---
Objects[] objects     # bounding box list for each of inferred images


================================================================================
MSG: object_msgs/Objects
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



(provide :object_msgs/ClassifyObject "530ab171b0ce2b4875431fb220cd7b7f")


