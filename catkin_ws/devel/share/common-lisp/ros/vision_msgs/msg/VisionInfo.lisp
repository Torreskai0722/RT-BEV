; Auto-generated. Do not edit!


(cl:in-package vision_msgs-msg)


;//! \htmlinclude VisionInfo.msg.html

(cl:defclass <VisionInfo> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (method
    :reader method
    :initarg :method
    :type cl:string
    :initform "")
   (database_location
    :reader database_location
    :initarg :database_location
    :type cl:string
    :initform "")
   (database_version
    :reader database_version
    :initarg :database_version
    :type cl:integer
    :initform 0))
)

(cl:defclass VisionInfo (<VisionInfo>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VisionInfo>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VisionInfo)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision_msgs-msg:<VisionInfo> is deprecated: use vision_msgs-msg:VisionInfo instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <VisionInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:header-val is deprecated.  Use vision_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'method-val :lambda-list '(m))
(cl:defmethod method-val ((m <VisionInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:method-val is deprecated.  Use vision_msgs-msg:method instead.")
  (method m))

(cl:ensure-generic-function 'database_location-val :lambda-list '(m))
(cl:defmethod database_location-val ((m <VisionInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:database_location-val is deprecated.  Use vision_msgs-msg:database_location instead.")
  (database_location m))

(cl:ensure-generic-function 'database_version-val :lambda-list '(m))
(cl:defmethod database_version-val ((m <VisionInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision_msgs-msg:database_version-val is deprecated.  Use vision_msgs-msg:database_version instead.")
  (database_version m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VisionInfo>) ostream)
  "Serializes a message object of type '<VisionInfo>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'method))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'method))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'database_location))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'database_location))
  (cl:let* ((signed (cl:slot-value msg 'database_version)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VisionInfo>) istream)
  "Deserializes a message object of type '<VisionInfo>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'method) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'method) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'database_location) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'database_location) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'database_version) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VisionInfo>)))
  "Returns string type for a message object of type '<VisionInfo>"
  "vision_msgs/VisionInfo")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VisionInfo)))
  "Returns string type for a message object of type 'VisionInfo"
  "vision_msgs/VisionInfo")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VisionInfo>)))
  "Returns md5sum for a message object of type '<VisionInfo>"
  "eee36f8dc558754ceb4ef619179d8b34")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VisionInfo)))
  "Returns md5sum for a message object of type 'VisionInfo"
  "eee36f8dc558754ceb4ef619179d8b34")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VisionInfo>)))
  "Returns full string definition for message of type '<VisionInfo>"
  (cl:format cl:nil "# Provides meta-information about a visual pipeline.~%#~%# This message serves a similar purpose to sensor_msgs/CameraInfo, but instead~%#   of being tied to hardware, it represents information about a specific~%#   computer vision pipeline. This information stays constant (or relatively~%#   constant) over time, and so it is wasteful to send it with each individual~%#   result. By listening to these messages, subscribers will receive~%#   the context in which published vision messages are to be interpreted.~%# Each vision pipeline should publish its VisionInfo messages to its own topic,~%#   in a manner similar to CameraInfo.~%~%# Used for sequencing~%Header header~%~%# Name of the vision pipeline. This should be a value that is meaningful to an~%#   outside user.~%string method~%~%# Location where the metadata database is stored. The recommended location is~%#   as an XML string on the ROS parameter server, but the exact implementation~%#   and information is left up to the user.~%# The database should store information attached to numeric ids. Each~%#   numeric id should map to an atomic, visually recognizable element. This~%#   definition is intentionally vague to allow extreme flexibility. The~%#   elements could be classes in a pixel segmentation algorithm, object classes~%#   in a detector, different people's faces in a face detection algorithm, etc.~%#   Vision pipelines report results in terms of numeric IDs, which map into~%#   this  database.~%# The information stored in this database is, again, left up to the user. The~%#   database could be as simple as a map from ID to class name, or it could~%#   include information such as object meshes or colors to use for~%#   visualization.~%string database_location~%~%# Metadata database version. This counter is incremented~%#   each time the pipeline begins using a new version of the database (useful~%#   in the case of online training or user modifications).~%#   The counter value can be monitored by listeners to ensure that the pipeline~%#   and the listener are using the same metadata.~%int32 database_version~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VisionInfo)))
  "Returns full string definition for message of type 'VisionInfo"
  (cl:format cl:nil "# Provides meta-information about a visual pipeline.~%#~%# This message serves a similar purpose to sensor_msgs/CameraInfo, but instead~%#   of being tied to hardware, it represents information about a specific~%#   computer vision pipeline. This information stays constant (or relatively~%#   constant) over time, and so it is wasteful to send it with each individual~%#   result. By listening to these messages, subscribers will receive~%#   the context in which published vision messages are to be interpreted.~%# Each vision pipeline should publish its VisionInfo messages to its own topic,~%#   in a manner similar to CameraInfo.~%~%# Used for sequencing~%Header header~%~%# Name of the vision pipeline. This should be a value that is meaningful to an~%#   outside user.~%string method~%~%# Location where the metadata database is stored. The recommended location is~%#   as an XML string on the ROS parameter server, but the exact implementation~%#   and information is left up to the user.~%# The database should store information attached to numeric ids. Each~%#   numeric id should map to an atomic, visually recognizable element. This~%#   definition is intentionally vague to allow extreme flexibility. The~%#   elements could be classes in a pixel segmentation algorithm, object classes~%#   in a detector, different people's faces in a face detection algorithm, etc.~%#   Vision pipelines report results in terms of numeric IDs, which map into~%#   this  database.~%# The information stored in this database is, again, left up to the user. The~%#   database could be as simple as a map from ID to class name, or it could~%#   include information such as object meshes or colors to use for~%#   visualization.~%string database_location~%~%# Metadata database version. This counter is incremented~%#   each time the pipeline begins using a new version of the database (useful~%#   in the case of online training or user modifications).~%#   The counter value can be monitored by listeners to ensure that the pipeline~%#   and the listener are using the same metadata.~%int32 database_version~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VisionInfo>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:length (cl:slot-value msg 'method))
     4 (cl:length (cl:slot-value msg 'database_location))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VisionInfo>))
  "Converts a ROS message object to a list"
  (cl:list 'VisionInfo
    (cl:cons ':header (header msg))
    (cl:cons ':method (method msg))
    (cl:cons ':database_location (database_location msg))
    (cl:cons ':database_version (database_version msg))
))
