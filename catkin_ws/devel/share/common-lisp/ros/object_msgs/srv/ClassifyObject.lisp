; Auto-generated. Do not edit!


(cl:in-package object_msgs-srv)


;//! \htmlinclude ClassifyObject-request.msg.html

(cl:defclass <ClassifyObject-request> (roslisp-msg-protocol:ros-message)
  ((image_paths
    :reader image_paths
    :initarg :image_paths
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element "")))
)

(cl:defclass ClassifyObject-request (<ClassifyObject-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ClassifyObject-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ClassifyObject-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_msgs-srv:<ClassifyObject-request> is deprecated: use object_msgs-srv:ClassifyObject-request instead.")))

(cl:ensure-generic-function 'image_paths-val :lambda-list '(m))
(cl:defmethod image_paths-val ((m <ClassifyObject-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_msgs-srv:image_paths-val is deprecated.  Use object_msgs-srv:image_paths instead.")
  (image_paths m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ClassifyObject-request>) ostream)
  "Serializes a message object of type '<ClassifyObject-request>"
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ClassifyObject-request>) istream)
  "Deserializes a message object of type '<ClassifyObject-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ClassifyObject-request>)))
  "Returns string type for a service object of type '<ClassifyObject-request>"
  "object_msgs/ClassifyObjectRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClassifyObject-request)))
  "Returns string type for a service object of type 'ClassifyObject-request"
  "object_msgs/ClassifyObjectRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ClassifyObject-request>)))
  "Returns md5sum for a message object of type '<ClassifyObject-request>"
  "530ab171b0ce2b4875431fb220cd7b7f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ClassifyObject-request)))
  "Returns md5sum for a message object of type 'ClassifyObject-request"
  "530ab171b0ce2b4875431fb220cd7b7f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ClassifyObject-request>)))
  "Returns full string definition for message of type '<ClassifyObject-request>"
  (cl:format cl:nil "# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string[] image_paths  # image path list for inference~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ClassifyObject-request)))
  "Returns full string definition for message of type 'ClassifyObject-request"
  (cl:format cl:nil "# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%string[] image_paths  # image path list for inference~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ClassifyObject-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'image_paths) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ClassifyObject-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ClassifyObject-request
    (cl:cons ':image_paths (image_paths msg))
))
;//! \htmlinclude ClassifyObject-response.msg.html

(cl:defclass <ClassifyObject-response> (roslisp-msg-protocol:ros-message)
  ((objects
    :reader objects
    :initarg :objects
    :type (cl:vector object_msgs-msg:Objects)
   :initform (cl:make-array 0 :element-type 'object_msgs-msg:Objects :initial-element (cl:make-instance 'object_msgs-msg:Objects))))
)

(cl:defclass ClassifyObject-response (<ClassifyObject-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ClassifyObject-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ClassifyObject-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_msgs-srv:<ClassifyObject-response> is deprecated: use object_msgs-srv:ClassifyObject-response instead.")))

(cl:ensure-generic-function 'objects-val :lambda-list '(m))
(cl:defmethod objects-val ((m <ClassifyObject-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_msgs-srv:objects-val is deprecated.  Use object_msgs-srv:objects instead.")
  (objects m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ClassifyObject-response>) ostream)
  "Serializes a message object of type '<ClassifyObject-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'objects))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'objects))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ClassifyObject-response>) istream)
  "Deserializes a message object of type '<ClassifyObject-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'objects) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'objects)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'object_msgs-msg:Objects))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ClassifyObject-response>)))
  "Returns string type for a service object of type '<ClassifyObject-response>"
  "object_msgs/ClassifyObjectResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClassifyObject-response)))
  "Returns string type for a service object of type 'ClassifyObject-response"
  "object_msgs/ClassifyObjectResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ClassifyObject-response>)))
  "Returns md5sum for a message object of type '<ClassifyObject-response>"
  "530ab171b0ce2b4875431fb220cd7b7f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ClassifyObject-response)))
  "Returns md5sum for a message object of type 'ClassifyObject-response"
  "530ab171b0ce2b4875431fb220cd7b7f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ClassifyObject-response>)))
  "Returns full string definition for message of type '<ClassifyObject-response>"
  (cl:format cl:nil "Objects[] objects     # bounding box list for each of inferred images~%~%~%================================================================================~%MSG: object_msgs/Objects~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent objects detected in a frame~%std_msgs/Header header    # timestamp in header is the time the sensor captured the raw data~%Object[] objects_vector   # Object array~%float32 inference_time_ms # inference time of this frame. the unit is millisecond.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: object_msgs/Object~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message define the property of detected object~%string object_name  # object name~%float32 probability # probability of detected object~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ClassifyObject-response)))
  "Returns full string definition for message of type 'ClassifyObject-response"
  (cl:format cl:nil "Objects[] objects     # bounding box list for each of inferred images~%~%~%================================================================================~%MSG: object_msgs/Objects~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message can represent objects detected in a frame~%std_msgs/Header header    # timestamp in header is the time the sensor captured the raw data~%Object[] objects_vector   # Object array~%float32 inference_time_ms # inference time of this frame. the unit is millisecond.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: object_msgs/Object~%# Copyright (c) 2017 Intel Corporation~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#      http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message define the property of detected object~%string object_name  # object name~%float32 probability # probability of detected object~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ClassifyObject-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'objects) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ClassifyObject-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ClassifyObject-response
    (cl:cons ':objects (objects msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ClassifyObject)))
  'ClassifyObject-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ClassifyObject)))
  'ClassifyObject-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClassifyObject)))
  "Returns string type for a service object of type '<ClassifyObject>"
  "object_msgs/ClassifyObject")