;; Auto-generated. Do not edit!


(when (boundp 'vision_msgs::BoundingBox3DArray)
  (if (not (find-package "VISION_MSGS"))
    (make-package "VISION_MSGS"))
  (shadow 'BoundingBox3DArray (find-package "VISION_MSGS")))
(unless (find-package "VISION_MSGS::BOUNDINGBOX3DARRAY")
  (make-package "VISION_MSGS::BOUNDINGBOX3DARRAY"))

(in-package "ROS")
;;//! \htmlinclude BoundingBox3DArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass vision_msgs::BoundingBox3DArray
  :super ros::object
  :slots (_header _boxes ))

(defmethod vision_msgs::BoundingBox3DArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:boxes __boxes) (let (r) (dotimes (i 0) (push (instance vision_msgs::BoundingBox3D :init) r)) r))
    )
   (send-super :init)
   (setq _header __header)
   (setq _boxes __boxes)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:boxes
   (&rest __boxes)
   (if (keywordp (car __boxes))
       (send* _boxes __boxes)
     (progn
       (if __boxes (setq _boxes (car __boxes)))
       _boxes)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; vision_msgs/BoundingBox3D[] _boxes
    (apply #'+ (send-all _boxes :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; vision_msgs/BoundingBox3D[] _boxes
     (write-long (length _boxes) s)
     (dolist (elem _boxes)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; vision_msgs/BoundingBox3D[] _boxes
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _boxes (let (r) (dotimes (i n) (push (instance vision_msgs::BoundingBox3D :init) r)) r))
     (dolist (elem- _boxes)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get vision_msgs::BoundingBox3DArray :md5sum-) "9e1a3932308592aa1b20232d818486db")
(setf (get vision_msgs::BoundingBox3DArray :datatype-) "vision_msgs/BoundingBox3DArray")
(setf (get vision_msgs::BoundingBox3DArray :definition-)
      "std_msgs/Header header
vision_msgs/BoundingBox3D[] boxes
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
MSG: vision_msgs/BoundingBox3D
# A 3D bounding box that can be positioned and rotated about its center (6 DOF)
# Dimensions of this box are in meters, and as such, it may be migrated to
#   another package, such as geometry_msgs, in the future.

# The 3D position and orientation of the bounding box center
geometry_msgs/Pose center

# The size of the bounding box, in meters, surrounding the object's center
#   pose.
geometry_msgs/Vector3 size

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :vision_msgs/BoundingBox3DArray "9e1a3932308592aa1b20232d818486db")


