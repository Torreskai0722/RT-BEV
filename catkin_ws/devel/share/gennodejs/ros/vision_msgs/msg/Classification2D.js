// Auto-generated. Do not edit!

// (in-package vision_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let ObjectHypothesis = require('./ObjectHypothesis.js');
let sensor_msgs = _finder('sensor_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Classification2D {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.results = null;
      this.source_img = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('results')) {
        this.results = initObj.results
      }
      else {
        this.results = [];
      }
      if (initObj.hasOwnProperty('source_img')) {
        this.source_img = initObj.source_img
      }
      else {
        this.source_img = new sensor_msgs.msg.Image();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Classification2D
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [results]
    // Serialize the length for message field [results]
    bufferOffset = _serializer.uint32(obj.results.length, buffer, bufferOffset);
    obj.results.forEach((val) => {
      bufferOffset = ObjectHypothesis.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [source_img]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.source_img, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Classification2D
    let len;
    let data = new Classification2D(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [results]
    // Deserialize array length for message field [results]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.results = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.results[i] = ObjectHypothesis.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [source_img]
    data.source_img = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 16 * object.results.length;
    length += sensor_msgs.msg.Image.getMessageSize(object.source_img);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'vision_msgs/Classification2D';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b23d0855d0f41568e09106615351255f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Defines a 2D classification result.
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Classification2D(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.results !== undefined) {
      resolved.results = new Array(msg.results.length);
      for (let i = 0; i < resolved.results.length; ++i) {
        resolved.results[i] = ObjectHypothesis.Resolve(msg.results[i]);
      }
    }
    else {
      resolved.results = []
    }

    if (msg.source_img !== undefined) {
      resolved.source_img = sensor_msgs.msg.Image.Resolve(msg.source_img)
    }
    else {
      resolved.source_img = new sensor_msgs.msg.Image()
    }

    return resolved;
    }
};

module.exports = Classification2D;
