;; Auto-generated. Do not edit!


(when (boundp 'vision_msgs::Classification2D)
  (if (not (find-package "VISION_MSGS"))
    (make-package "VISION_MSGS"))
  (shadow 'Classification2D (find-package "VISION_MSGS")))
(unless (find-package "VISION_MSGS::CLASSIFICATION2D")
  (make-package "VISION_MSGS::CLASSIFICATION2D"))

(in-package "ROS")
;;//! \htmlinclude Classification2D.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass vision_msgs::Classification2D
  :super ros::object
  :slots (_header _results _source_img ))

(defmethod vision_msgs::Classification2D
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:results __results) (let (r) (dotimes (i 0) (push (instance vision_msgs::ObjectHypothesis :init) r)) r))
    ((:source_img __source_img) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _results __results)
   (setq _source_img __source_img)
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
  (:source_img
   (&rest __source_img)
   (if (keywordp (car __source_img))
       (send* _source_img __source_img)
     (progn
       (if __source_img (setq _source_img (car __source_img)))
       _source_img)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; vision_msgs/ObjectHypothesis[] _results
    (apply #'+ (send-all _results :serialization-length)) 4
    ;; sensor_msgs/Image _source_img
    (send _source_img :serialization-length)
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
     ;; sensor_msgs/Image _source_img
       (send _source_img :serialize s)
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
   ;; sensor_msgs/Image _source_img
     (send _source_img :deserialize buf ptr-) (incf ptr- (send _source_img :serialization-length))
   ;;
   self)
  )

(setf (get vision_msgs::Classification2D :md5sum-) "b23d0855d0f41568e09106615351255f")
(setf (get vision_msgs::Classification2D :datatype-) "vision_msgs/Classification2D")
(setf (get vision_msgs::Classification2D :definition-)
      "# Defines a 2D classification result.
#
# This result does not contain any position information. It is designed for
#   classifiers, which simply provide class probabilities given a source image.

Header header

# A list of class probabilities. This list need not provide a probability for
#   every possible class, just ones that are nonzero, or above some
#   user-defined threshold.
ObjectHypothesis[] results

# The 2D data that generated these results (i.e. region proposal cropped out of
#   the image). Not required for all use cases, so it may be empty.
sensor_msgs/Image source_img
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
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

")



(provide :vision_msgs/Classification2D "b23d0855d0f41568e09106615351255f")


