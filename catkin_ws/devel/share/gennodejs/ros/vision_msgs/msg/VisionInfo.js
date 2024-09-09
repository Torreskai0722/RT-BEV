// Auto-generated. Do not edit!

// (in-package vision_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class VisionInfo {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.method = null;
      this.database_location = null;
      this.database_version = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('method')) {
        this.method = initObj.method
      }
      else {
        this.method = '';
      }
      if (initObj.hasOwnProperty('database_location')) {
        this.database_location = initObj.database_location
      }
      else {
        this.database_location = '';
      }
      if (initObj.hasOwnProperty('database_version')) {
        this.database_version = initObj.database_version
      }
      else {
        this.database_version = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type VisionInfo
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [method]
    bufferOffset = _serializer.string(obj.method, buffer, bufferOffset);
    // Serialize message field [database_location]
    bufferOffset = _serializer.string(obj.database_location, buffer, bufferOffset);
    // Serialize message field [database_version]
    bufferOffset = _serializer.int32(obj.database_version, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type VisionInfo
    let len;
    let data = new VisionInfo(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [method]
    data.method = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [database_location]
    data.database_location = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [database_version]
    data.database_version = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.method.length;
    length += object.database_location.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'vision_msgs/VisionInfo';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eee36f8dc558754ceb4ef619179d8b34';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Provides meta-information about a visual pipeline.
    #
    # This message serves a similar purpose to sensor_msgs/CameraInfo, but instead
    #   of being tied to hardware, it represents information about a specific
    #   computer vision pipeline. This information stays constant (or relatively
    #   constant) over time, and so it is wasteful to send it with each individual
    #   result. By listening to these messages, subscribers will receive
    #   the context in which published vision messages are to be interpreted.
    # Each vision pipeline should publish its VisionInfo messages to its own topic,
    #   in a manner similar to CameraInfo.
    
    # Used for sequencing
    Header header
    
    # Name of the vision pipeline. This should be a value that is meaningful to an
    #   outside user.
    string method
    
    # Location where the metadata database is stored. The recommended location is
    #   as an XML string on the ROS parameter server, but the exact implementation
    #   and information is left up to the user.
    # The database should store information attached to numeric ids. Each
    #   numeric id should map to an atomic, visually recognizable element. This
    #   definition is intentionally vague to allow extreme flexibility. The
    #   elements could be classes in a pixel segmentation algorithm, object classes
    #   in a detector, different people's faces in a face detection algorithm, etc.
    #   Vision pipelines report results in terms of numeric IDs, which map into
    #   this  database.
    # The information stored in this database is, again, left up to the user. The
    #   database could be as simple as a map from ID to class name, or it could
    #   include information such as object meshes or colors to use for
    #   visualization.
    string database_location
    
    # Metadata database version. This counter is incremented
    #   each time the pipeline begins using a new version of the database (useful
    #   in the case of online training or user modifications).
    #   The counter value can be monitored by listeners to ensure that the pipeline
    #   and the listener are using the same metadata.
    int32 database_version
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new VisionInfo(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.method !== undefined) {
      resolved.method = msg.method;
    }
    else {
      resolved.method = ''
    }

    if (msg.database_location !== undefined) {
      resolved.database_location = msg.database_location;
    }
    else {
      resolved.database_location = ''
    }

    if (msg.database_version !== undefined) {
      resolved.database_version = msg.database_version;
    }
    else {
      resolved.database_version = 0
    }

    return resolved;
    }
};

module.exports = VisionInfo;
