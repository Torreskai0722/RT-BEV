;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::ObjectInBox)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'ObjectInBox (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::OBJECTINBOX")
  (make-package "OBJECT_MSGS::OBJECTINBOX"))

(in-package "ROS")
;;//! \htmlinclude ObjectInBox.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass object_msgs::ObjectInBox
  :super ros::object
  :slots (_object _roi ))

(defmethod object_msgs::ObjectInBox
  (:init
   (&key
    ((:object __object) (instance object_msgs::Object :init))
    ((:roi __roi) (instance sensor_msgs::RegionOfInterest :init))
    )
   (send-super :init)
   (setq _object __object)
   (setq _roi __roi)
   self)
  (:object
   (&rest __object)
   (if (keywordp (car __object))
       (send* _object __object)
     (progn
       (if __object (setq _object (car __object)))
       _object)))
  (:roi
   (&rest __roi)
   (if (keywordp (car __roi))
       (send* _roi __roi)
     (progn
       (if __roi (setq _roi (car __roi)))
       _roi)))
  (:serialization-length
   ()
   (+
    ;; object_msgs/Object _object
    (send _object :serialization-length)
    ;; sensor_msgs/RegionOfInterest _roi
    (send _roi :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; object_msgs/Object _object
       (send _object :serialize s)
     ;; sensor_msgs/RegionOfInterest _roi
       (send _roi :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; object_msgs/Object _object
     (send _object :deserialize buf ptr-) (incf ptr- (send _object :serialization-length))
   ;; sensor_msgs/RegionOfInterest _roi
     (send _roi :deserialize buf ptr-) (incf ptr- (send _roi :serialization-length))
   ;;
   self)
  )

(setf (get object_msgs::ObjectInBox :md5sum-) "658b4f57dcac376ed1902dc618650a32")
(setf (get object_msgs::ObjectInBox :datatype-) "object_msgs/ObjectInBox")
(setf (get object_msgs::ObjectInBox :definition-)
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



(provide :object_msgs/ObjectInBox "658b4f57dcac376ed1902dc618650a32")


