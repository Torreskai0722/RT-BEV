;; Auto-generated. Do not edit!


(when (boundp 'vision_msgs::Classification3D)
  (if (not (find-package "VISION_MSGS"))
    (make-package "VISION_MSGS"))
  (shadow 'Classification3D (find-package "VISION_MSGS")))
(unless (find-package "VISION_MSGS::CLASSIFICATION3D")
  (make-package "VISION_MSGS::CLASSIFICATION3D"))

(in-package "ROS")
;;//! \htmlinclude Classification3D.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass vision_msgs::Classification3D
  :super ros::object
  :slots (_header _results _source_cloud ))

(defmethod vision_msgs::Classification3D
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:results __results) (let (r) (dotimes (i 0) (push (instance vision_msgs::ObjectHypothesis :init) r)) r))
    ((:source_cloud __source_cloud) (instance sensor_msgs::PointCloud2 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _results __results)
   (setq _source_cloud __source_cloud)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:results
   (&rest __results)
   (if (keywordp (car __results))
       (send* _results __results)
     (progn
       (if __results (setq _results (car __results)))
       _results)))
  (:source_cloud
   (&rest __source_cloud)
   (if (keywordp (car __source_cloud))
       (send* _source_cloud __source_cloud)
     (progn
       (if __source_cloud (setq _source_cloud (car __source_cloud)))
       _source_cloud)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; vision_msgs/ObjectHypothesis[] _results
    (apply #'+ (send-all _results :serialization-length)) 4
    ;; sensor_msgs/PointCloud2 _source_cloud
    (send _source_cloud :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; vision_msgs/ObjectHypothesis[] _results
     (write-long (length _results) s)
     (dolist (elem _results)
       (send elem :serialize s)
       )
     ;; sensor_msgs/PointCloud2 _source_cloud
       (send _source_cloud :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; vision_msgs/ObjectHypothesis[] _results
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _results (let (r) (dotimes (i n) (push (instance vision_msgs::ObjectHypothesis :init) r)) r))
     (dolist (elem- _results)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; sensor_msgs/PointCloud2 _source_cloud
     (send _source_cloud :deserialize buf ptr-) (incf ptr- (send _source_cloud :serialization-length))
   ;;
   self)
  )

(setf (get vision_msgs::Classification3D :md5sum-) "2c0fe97799b60ee2995363b3fbf44715")
(setf (get vision_msgs::Classification3D :datatype-) "vision_msgs/Classification3D")
(setf (get vision_msgs::Classification3D :definition-)
      "# Defines a 3D classification result.
#
# This result does not contain any position information. It is designed for
#   classifiers, which simply provide probabilities given a source image.

Header header

# Class probabilities
ObjectHypothesis[] results

# The 3D data that generated these results (i.e. region proposal cropped out of
#   the image). Not required for all detectors, so it may be empty.
sensor_msgs/PointCloud2 source_cloud
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
MSG: vision_msgs/ObjectHypothesis
# An object hypothesis that contains no position information.

# The unique numeric ID of object detected. To get additional information about
#   this ID, such as its human-readable name, listeners should perform a lookup
#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.
int64 id

# The probability or confidence value of the detected object. By convention,
#   this value should lie in the range [0-1].
float64 score
================================================================================
MSG: sensor_msgs/PointCloud2
# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its layout described by the
# contents of the \"fields\" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
Header header

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points

================================================================================
MSG: sensor_msgs/PointField
# This message holds the description of one point entry in the
# PointCloud2 message format.
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8

string name      # Name of field
uint32 offset    # Offset from start of point struct
uint8  datatype  # Datatype enumeration, see above
uint32 count     # How many elements in the field

")



(provide :vision_msgs/Classification3D "2c0fe97799b60ee2995363b3fbf44715")


