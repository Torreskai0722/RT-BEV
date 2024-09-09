; Auto-generated. Do not edit!


(cl:in-package vision_msgs-msg)


;//! \htmlinclude Classification3D.msg.html

(cl:defclass <Classification3D> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (results
    :reader results
    :initarg :results
    :type (cl:vector vision_msgs-msg:ObjectHypothesis)
   :initform (cl:make-array 0 :element-type 'vision_msgs-msg:ObjectHypothesis :initial-element (cl:make-instance 'vision_msgs-msg:ObjectHypothesis)))
   (source_cloud
    :reader source_cloud
    :initarg :source_cloud
    :type sensor_msgs-msg:PointCloud2
    :initform (cl:make-instance 'sensor_msgs-msg:PointCloud2)))
)

(cl:defclass Classification3D (<Classification3D>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Classification3D>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Classification3D)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision_msgs-msg:<Classification3D> is deprecated: use vision_msgs-msg:Classification3D instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Classification3D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:header-val is deprecated.  Use vision_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'results-val :lambda-list '(m))
(cl:defmethod results-val ((m <Classification3D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:results-val is deprecated.  Use vision_msgs-msg:results instead.")
  (results m))

(cl:ensure-generic-function 'source_cloud-val :lambda-list '(m))
(cl:defmethod source_cloud-val ((m <Classification3D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:source_cloud-val is deprecated.  Use vision_msgs-msg:source_cloud instead.")
  (source_cloud m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Classification3D>) ostream)
  "Serializes a message object of type '<Classification3D>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'results))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'results))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'source_cloud) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Classification3D>) istream)
  "Deserializes a message object of type '<Classification3D>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'results) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'results)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'vision_msgs-msg:ObjectHypothesis))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'source_cloud) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Classification3D>)))
  "Returns string type for a message object of type '<Classification3D>"
  "vision_msgs/Classification3D")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Classification3D)))
  "Returns string type for a message object of type 'Classification3D"
  "vision_msgs/Classification3D")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Classification3D>)))
  "Returns md5sum for a message object of type '<Classification3D>"
  "2c0fe97799b60ee2995363b3fbf44715")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Classification3D)))
  "Returns md5sum for a message object of type 'Classification3D"
  "2c0fe97799b60ee2995363b3fbf44715")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Classification3D>)))
  "Returns full string definition for message of type '<Classification3D>"
  (cl:format cl:nil "# Defines a 3D classification result.~%#~%# This result does not contain any position information. It is designed for~%#   classifiers, which simply provide probabilities given a source image.~%~%Header header~%~%# Class probabilities~%ObjectHypothesis[] results~%~%# The 3D data that generated these results (i.e. region proposal cropped out of~%#   the image). Not required for all detectors, so it may be empty.~%sensor_msgs/PointCloud2 source_cloud~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: vision_msgs/ObjectHypothesis~%# An object hypothesis that contains no position information.~%~%# The unique numeric ID of object detected. To get additional information about~%#   this ID, such as its human-readable name, listeners should perform a lookup~%#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.~%int64 id~%~%# The probability or confidence value of the detected object. By convention,~%#   this value should lie in the range [0-1].~%float64 score~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Classification3D)))
  "Returns full string definition for message of type 'Classification3D"
  (cl:format cl:nil "# Defines a 3D classification result.~%#~%# This result does not contain any position information. It is designed for~%#   classifiers, which simply provide probabilities given a source image.~%~%Header header~%~%# Class probabilities~%ObjectHypothesis[] results~%~%# The 3D data that generated these results (i.e. region proposal cropped out of~%#   the image). Not required for all detectors, so it may be empty.~%sensor_msgs/PointCloud2 source_cloud~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: vision_msgs/ObjectHypothesis~%# An object hypothesis that contains no position information.~%~%# The unique numeric ID of object detected. To get additional information about~%#   this ID, such as its human-readable name, listeners should perform a lookup~%#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.~%int64 id~%~%# The probability or confidence value of the detected object. By convention,~%#   this value should lie in the range [0-1].~%float64 score~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Classification3D>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'results) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'source_cloud))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Classification3D>))
  "Converts a ROS message object to a list"
  (cl:list 'Classification3D
    (cl:cons ':header (header msg))
    (cl:cons ':results (results msg))
    (cl:cons ':source_cloud (source_cloud msg))
))
