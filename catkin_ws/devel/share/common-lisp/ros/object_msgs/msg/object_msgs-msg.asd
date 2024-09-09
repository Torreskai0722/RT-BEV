
(cl:in-package :asdf)

(defsystem "object_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Object" :depends-on ("_package_Object"))
    (:file "_package_Object" :depends-on ("_package"))
    (:file "ObjectInBox" :depends-on ("_package_ObjectInBox"))
    (:file "_package_ObjectInBox" :depends-on ("_package"))
    (:file "Objects" :depends-on ("_package_Objects"))
    (:file "_package_Objects" :depends-on ("_package"))
    (:file "ObjectsInBoxes" :depends-on ("_package_ObjectsInBoxes"))
    (:file "_package_ObjectsInBoxes" :depends-on ("_package"))
  ))