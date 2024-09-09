; Auto-generated. Do not edit!


(cl:in-package object_msgs-srv)


;//! \htmlinclude DetectObject-request.msg.html

(cl:defclass <DetectObject-request> (roslisp-msg-protocol:ros-message)
  ((image_paths
    :reader image_paths
    :initarg :image_paths
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element "")))
)

(cl:defclass DetectObject-request (<DetectObject-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectObject-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectObject-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_msgs-srv:<DetectObject-request> is deprecated: use object_msgs-srv:DetectObject-request instead.")))

(cl:ensure-generic-function 'image_paths-val :lambda-list '(m))
(cl:defmethod image_paths-val ((m <DetectObject-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_msgs-srv:image_paths-val is deprecated.  Use object_msgs-srv:image_paths instead.")
  (image_paths m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectObject-request>) ostream)
  "Serializes a message object of type '<DetectObject-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'image_paths))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'image_paths))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectObject-request>) istream)
  "Deserializes a message object of type '<DetectObject-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'image_paths) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'image_paths)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectObject-request>)))
  "Returns string type for a service object of type '<DetectObject-request>"
  "object_msgs/DetectObjectRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObject-request)))
  "Returns string type for a service object of type 'DetectObject-request"
  "object_msgs/DetectObjectRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectObject-request>)))
  "Returns md5sum for a message object of type '<DetectObject-request>"
  "11b2ac83822624e9953f8718798e5220")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectObject-request)))
  "Returns md5sum for a message object of type 'DetectObject-request"
  "11b2ac83822624e9953f8718798e5220")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectObject-request>)))
  "Returns full string definition for message of type '<DetectObject-request>"
  (cl:format cl:nil "# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string[] image_paths      # image path list for inference~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectObject-request)))
  "Returns full string definition for message of type 'DetectObject-request"
  (cl:format cl:nil "# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string[] image_paths      # image path list for inference~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectObject-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'image_paths) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectObject-request>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectObject-request
    (cl:cons ':image_paths (image_paths msg))
))
;//! \htmlinclude DetectObject-response.msg.html

(cl:defclass <DetectObject-response> (roslisp-msg-protocol:ros-message)
  ((objects
    :reader objects
    :initarg :objects
    :type (cl:vector object_msgs-msg:ObjectsInBoxes)
   :initform (cl:make-array 0 :element-type 'object_msgs-msg:ObjectsInBoxes :initial-element (cl:make-instance 'object_msgs-msg:ObjectsInBoxes))))
)

(cl:defclass DetectObject-response (<DetectObject-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectObject-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectObject-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_msgs-srv:<DetectObject-response> is deprecated: use object_msgs-srv:DetectObject-response instead.")))

(cl:ensure-generic-function 'objects-val :lambda-list '(m))
(cl:defmethod objects-val ((m <DetectObject-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_msgs-srv:objects-val is deprecated.  Use object_msgs-srv:objects instead.")
  (objects m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectObject-response>) ostream)
  "Serializes a message object of type '<DetectObject-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'objects))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'objects))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectObject-response>) istream)
  "Deserializes a message object of type '<DetectObject-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'objects) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'objects)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'object_msgs-msg:ObjectsInBoxes))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectObject-response>)))
  "Returns string type for a service object of type '<DetectObject-response>"
  "object_msgs/DetectObjectResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObject-response)))
  "Returns string type for a service object of type 'DetectObject-response"
  "object_msgs/DetectObjectResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectObject-response>)))
  "Returns md5sum for a message object of type '<DetectObject-response>"
  "11b2ac83822624e9953f8718798e5220")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectObject-response)))
  "Returns md5sum for a message object of type 'DetectObject-response"
  "11b2ac83822624e9953f8718798e5220")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectObject-response>)))
  "Returns full string definition for message of type '<DetectObject-response>"
  (cl:format cl:nil "ObjectsInBoxes[] objects  # bounding box list for each of inferred images~%~%~%================================================================================~%MSG: object_msgs/ObjectsInBoxes~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent objects detected and their bounding-boxes in a frame~%std_msgs/Header header        # timestamp in header is the time the sensor captured the raw data~%ObjectInBox[] objects_vector  # ObjectInBox array~%float32 inference_time_ms     # inference time of this frame. the unit is millisecond.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: object_msgs/ObjectInBox~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent a detected object and its region of interest~%Object object                     # detected object~%sensor_msgs/RegionOfInterest roi  # region of interest~%~%================================================================================~%MSG: object_msgs/Object~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message define the property of detected object~%string object_name  # object name~%float32 probability # probability of detected object~%~%================================================================================~%MSG: sensor_msgs/RegionOfInterest~%# This message is used to specify a region of interest within an image.~%#~%# When used to specify the ROI setting of the camera when the image was~%# taken, the height and width fields should either match the height and~%# width fields for the associated image; or height = width = 0~%# indicates that the full resolution image was captured.~%~%uint32 x_offset  # Leftmost pixel of the ROI~%                 # (0 if the ROI includes the left edge of the image)~%uint32 y_offset  # Topmost pixel of the ROI~%                 # (0 if the ROI includes the top edge of the image)~%uint32 height    # Height of ROI~%uint32 width     # Width of ROI~%~%# True if a distinct rectified ROI should be calculated from the \"raw\"~%# ROI in this message. Typically this should be False if the full image~%# is captured (ROI not used), and True if a subwindow is captured (ROI~%# used).~%bool do_rectify~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectObject-response)))
  "Returns full string definition for message of type 'DetectObject-response"
  (cl:format cl:nil "ObjectsInBoxes[] objects  # bounding box list for each of inferred images~%~%~%================================================================================~%MSG: object_msgs/ObjectsInBoxes~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent objects detected and their bounding-boxes in a frame~%std_msgs/Header header        # timestamp in header is the time the sensor captured the raw data~%ObjectInBox[] objects_vector  # ObjectInBox array~%float32 inference_time_ms     # inference time of this frame. the unit is millisecond.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: object_msgs/ObjectInBox~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent a detected object and its region of interest~%Object object                     # detected object~%sensor_msgs/RegionOfInterest roi  # region of interest~%~%================================================================================~%MSG: object_msgs/Object~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message define the property of detected object~%string object_name  # object name~%float32 probability # probability of detected object~%~%================================================================================~%MSG: sensor_msgs/RegionOfInterest~%# This message is used to specify a region of interest within an image.~%#~%# When used to specify the ROI setting of the camera when the image was~%# taken, the height and width fields should either match the height and~%# width fields for the associated image; or height = width = 0~%# indicates that the full resolution image was captured.~%~%uint32 x_offset  # Leftmost pixel of the ROI~%                 # (0 if the ROI includes the left edge of the image)~%uint32 y_offset  # Topmost pixel of the ROI~%                 # (0 if the ROI includes the top edge of the image)~%uint32 height    # Height of ROI~%uint32 width     # Width of ROI~%~%# True if a distinct rectified ROI should be calculated from the \"raw\"~%# ROI in this message. Typically this should be False if the full image~%# is captured (ROI not used), and True if a subwindow is captured (ROI~%# used).~%bool do_rectify~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectObject-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'objects) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectObject-response>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectObject-response
    (cl:cons ':objects (objects msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'DetectObject)))
  'DetectObject-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'DetectObject)))
  'DetectObject-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObject)))
  "Returns string type for a service object of type '<DetectObject>"
  "object_msgs/DetectObject")