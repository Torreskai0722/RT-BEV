// Auto-generated. Do not edit!

// (in-package object_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let Object = require('./Object.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Objects {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.objects_vector = null;
      this.inference_time_ms = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('objects_vector')) {
        this.objects_vector = initObj.objects_vector
      }
      else {
        this.objects_vector = [];
      }
      if (initObj.hasOwnProperty('inference_time_ms')) {
        this.inference_time_ms = initObj.inference_time_ms
      }
      else {
        this.inference_time_ms = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Objects
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [objects_vector]
    // Serialize the length for message field [objects_vector]
    bufferOffset = _serializer.uint32(obj.objects_vector.length, buffer, bufferOffset);
    obj.objects_vector.forEach((val) => {
      bufferOffset = Object.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [inference_time_ms]
    bufferOffset = _serializer.float32(obj.inference_time_ms, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Objects
    let len;
    let data = new Objects(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [objects_vector]
    // Deserialize array length for message field [objects_vector]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.objects_vector = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.objects_vector[i] = Object.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [inference_time_ms]
    data.inference_time_ms = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.objects_vector.forEach((val) => {
      length += Object.getMessageSize(val);
    });
    return length + 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'object_msgs/Objects';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '491e959c05e1aac76496e491368f3ef1';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Copyright (c) 2017 Intel Corporation
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # This message can represent objects detected in a frame
    std_msgs/Header header    # timestamp in header is the time the sensor captured the raw data
    Object[] objects_vector   # Object array
    float32 inference_time_ms # inference time of this frame. the unit is millisecond.
    
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
    MSG: object_msgs/Object
    # Copyright (c) 2017 Intel Corporation
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # This message define the property of detected object
    string object_name  # object name
    float32 probability # probability of detected object
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Objects(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.objects_vector !== undefined) {
      resolved.objects_vector = new Array(msg.objects_vector.length);
      for (let i = 0; i < resolved.objects_vector.length; ++i) {
        resolved.objects_vector[i] = Object.Resolve(msg.objects_vector[i]);
      }
    }
    else {
      resolved.objects_vector = []
    }

    if (msg.inference_time_ms !== undefined) {
      resolved.inference_time_ms = msg.inference_time_ms;
    }
    else {
      resolved.inference_time_ms = 0.0
    }

    return resolved;
    }
};

module.exports = Objects;
