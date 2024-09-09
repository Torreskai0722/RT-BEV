;; Auto-generated. Do not edit!


(when (boundp 'vision_msgs::VisionInfo)
  (if (not (find-package "VISION_MSGS"))
    (make-package "VISION_MSGS"))
  (shadow 'VisionInfo (find-package "VISION_MSGS")))
(unless (find-package "VISION_MSGS::VISIONINFO")
  (make-package "VISION_MSGS::VISIONINFO"))

(in-package "ROS")
;;//! \htmlinclude VisionInfo.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass vision_msgs::VisionInfo
  :super ros::object
  :slots (_header _method _database_location _database_version ))

(defmethod vision_msgs::VisionInfo
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:method __method) "")
    ((:database_location __database_location) "")
    ((:database_version __database_version) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _method (string __method))
   (setq _database_location (string __database_location))
   (setq _database_version (round __database_version))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:method
   (&optional __method)
   (if __method (setq _method __method)) _method)
  (:database_location
   (&optional __database_location)
   (if __database_location (setq _database_location __database_location)) _database_location)
  (:database_version
   (&optional __database_version)
   (if __database_version (setq _database_version __database_version)) _database_version)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _method
    4 (length _method)
    ;; string _database_location
    4 (length _database_location)
    ;; int32 _database_version
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _method
       (write-long (length _method) s) (princ _method s)
     ;; string _database_location
       (write-long (length _database_location) s) (princ _database_location s)
     ;; int32 _database_version
       (write-long _database_version s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _method
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _method (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _database_location
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _database_location (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; int32 _database_version
     (setq _database_version (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get vision_msgs::VisionInfo :md5sum-) "eee36f8dc558754ceb4ef619179d8b34")
(setf (get vision_msgs::VisionInfo :datatype-) "vision_msgs/VisionInfo")
(setf (get vision_msgs::VisionInfo :definition-)
      "# Provides meta-information about a visual pipeline.
#
# This message serves a similar purpose to sensor_msgs/CameraInfo, but instead
#   of being tied to hardware, it represents information about a specific
#   computer vision pipeline. This information stays constant (or relatively
#   constant) over time, and so it is wasteful to send it with each individual
#   result. By listening to these messages, subscribers will receive
#   the context in which published vision messages are to be interpreted.
# Each vision pipeline should publish its VisionInfo messages to its own topic,
#   in a manner similar to CameraInfo.

# Used for sequencing
Header header

# Name of the vision pipeline. This should be a value that is meaningful to an
#   outside user.
string method

# Location where the metadata database is stored. The recommended location is
#   as an XML string on the ROS parameter server, but the exact implementation
#   and information is left up to the user.
# The database should store information attached to numeric ids. Each
#   numeric id should map to an atomic, visually recognizable element. This
#   definition is intentionally vague to allow extreme flexibility. The
#   elements could be classes in a pixel segmentation algorithm, object classes
#   in a detector, different people's faces in a face detection algorithm, etc.
#   Vision pipelines report results in terms of numeric IDs, which map into
#   this  database.
# The information stored in this database is, again, left up to the user. The
#   database could be as simple as a map from ID to class name, or it could
#   include information such as object meshes or colors to use for
#   visualization.
string database_location

# Metadata database version. This counter is incremented
#   each time the pipeline begins using a new version of the database (useful
#   in the case of online training or user modifications).
#   The counter value can be monitored by listeners to ensure that the pipeline
#   and the listener are using the same metadata.
int32 database_version
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

")



(provide :vision_msgs/VisionInfo "eee36f8dc558754ceb4ef619179d8b34")


