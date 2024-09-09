;; Auto-generated. Do not edit!


(when (boundp 'object_msgs::Object)
  (if (not (find-package "OBJECT_MSGS"))
    (make-package "OBJECT_MSGS"))
  (shadow 'Object (find-package "OBJECT_MSGS")))
(unless (find-package "OBJECT_MSGS::OBJECT")
  (make-package "OBJECT_MSGS::OBJECT"))

(in-package "ROS")
;;//! \htmlinclude Object.msg.html


(defclass object_msgs::Object
  :super ros::object
  :slots (_object_name _probability ))

(defmethod object_msgs::Object
  (:init
   (&key
    ((:object_name __object_name) "")
    ((:probability __probability) 0.0)
    )
   (send-super :init)
   (setq _object_name (string __object_name))
   (setq _probability (float __probability))
   self)
  (:object_name
   (&optional __object_name)
   (if __object_name (setq _object_name __object_name)) _object_name)
  (:probability
   (&optional __probability)
   (if __probability (setq _probability __probability)) _probability)
  (:serialization-length
   ()
   (+
    ;; string _object_name
    4 (length _object_name)
    ;; float32 _probability
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _object_name
       (write-long (length _object_name) s) (princ _object_name s)
     ;; float32 _probability
       (sys::poke _probability (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _object_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _object_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _probability
     (setq _probability (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get object_msgs::Object :md5sum-) "b62386628eb32e68aec0dfe4b39247d3")
(setf (get object_msgs::Object :datatype-) "object_msgs/Object")
(setf (get object_msgs::Object :definition-)
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

# This message define the property of detected object
string object_name  # object name
float32 probability # probability of detected object

")



(provide :object_msgs/Object "b62386628eb32e68aec0dfe4b39247d3")


