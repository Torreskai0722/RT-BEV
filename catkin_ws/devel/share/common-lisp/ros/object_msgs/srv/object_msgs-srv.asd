
(cl:in-package :asdf)

(defsystem "object_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :object_msgs-msg
)
  :components ((:file "_package")
    (:file "ClassifyObject" :depends-on ("_package_ClassifyObject"))
    (:file "_package_ClassifyObject" :depends-on ("_package"))
    (:file "DetectObject" :depends-on ("_package_DetectObject"))
    (:file "_package_DetectObject" :depends-on ("_package"))
  ))