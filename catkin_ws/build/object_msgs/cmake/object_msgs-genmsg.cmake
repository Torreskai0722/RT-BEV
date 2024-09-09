# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "object_msgs: 4 messages, 2 services")

set(MSG_I_FLAGS "-Iobject_msgs:/home/mobilitylab/catkin_ws/src/object_msgs/msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/melodic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(object_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" ""
)

get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" "std_msgs/Header:object_msgs/Object"
)

get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" "sensor_msgs/RegionOfInterest:object_msgs/Object"
)

get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" "sensor_msgs/RegionOfInterest:object_msgs/ObjectInBox:std_msgs/Header:object_msgs/Object"
)

get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" "object_msgs/Objects:std_msgs/Header:object_msgs/Object"
)

get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_custom_target(_object_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "object_msgs" "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" "std_msgs/Header:sensor_msgs/RegionOfInterest:object_msgs/ObjectsInBoxes:object_msgs/ObjectInBox:object_msgs/Object"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)
_generate_msg_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)
_generate_msg_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)
_generate_msg_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)

### Generating Services
_generate_srv_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv"
  "${MSG_I_FLAGS}"
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)
_generate_srv_cpp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
)

### Generating Module File
_generate_module_cpp(object_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(object_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(object_msgs_generate_messages object_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_cpp _object_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(object_msgs_gencpp)
add_dependencies(object_msgs_gencpp object_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS object_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)
_generate_msg_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)
_generate_msg_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)
_generate_msg_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)

### Generating Services
_generate_srv_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv"
  "${MSG_I_FLAGS}"
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)
_generate_srv_eus(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
)

### Generating Module File
_generate_module_eus(object_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(object_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(object_msgs_generate_messages object_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_eus _object_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(object_msgs_geneus)
add_dependencies(object_msgs_geneus object_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS object_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)
_generate_msg_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)
_generate_msg_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)
_generate_msg_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)

### Generating Services
_generate_srv_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv"
  "${MSG_I_FLAGS}"
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)
_generate_srv_lisp(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
)

### Generating Module File
_generate_module_lisp(object_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(object_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(object_msgs_generate_messages object_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_lisp _object_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(object_msgs_genlisp)
add_dependencies(object_msgs_genlisp object_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS object_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)
_generate_msg_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)
_generate_msg_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)
_generate_msg_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)

### Generating Services
_generate_srv_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv"
  "${MSG_I_FLAGS}"
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)
_generate_srv_nodejs(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
)

### Generating Module File
_generate_module_nodejs(object_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(object_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(object_msgs_generate_messages object_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_nodejs _object_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(object_msgs_gennodejs)
add_dependencies(object_msgs_gennodejs object_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS object_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)
_generate_msg_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)
_generate_msg_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)
_generate_msg_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)

### Generating Services
_generate_srv_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv"
  "${MSG_I_FLAGS}"
  "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)
_generate_srv_py(object_msgs
  "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/melodic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg;/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
)

### Generating Module File
_generate_module_py(object_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(object_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(object_msgs_generate_messages object_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Object.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/Objects.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectInBox.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/msg/ObjectsInBoxes.msg" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/ClassifyObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mobilitylab/catkin_ws/src/object_msgs/srv/DetectObject.srv" NAME_WE)
add_dependencies(object_msgs_generate_messages_py _object_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(object_msgs_genpy)
add_dependencies(object_msgs_genpy object_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS object_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/object_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(object_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(object_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/object_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(object_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(object_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/object_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(object_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(object_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/object_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(object_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(object_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/object_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(object_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(object_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
