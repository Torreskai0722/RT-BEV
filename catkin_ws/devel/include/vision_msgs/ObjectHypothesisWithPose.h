// Generated by gencpp from file vision_msgs/ObjectHypothesisWithPose.msg
// DO NOT EDIT!


#ifndef VISION_MSGS_MESSAGE_OBJECTHYPOTHESISWITHPOSE_H
#define VISION_MSGS_MESSAGE_OBJECTHYPOTHESISWITHPOSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/PoseWithCovariance.h>

namespace vision_msgs
{
template <class ContainerAllocator>
struct ObjectHypothesisWithPose_
{
  typedef ObjectHypothesisWithPose_<ContainerAllocator> Type;

  ObjectHypothesisWithPose_()
    : id(0)
    , score(0.0)
    , pose()  {
    }
  ObjectHypothesisWithPose_(const ContainerAllocator& _alloc)
    : id(0)
    , score(0.0)
    , pose(_alloc)  {
  (void)_alloc;
    }



   typedef int64_t _id_type;
  _id_type id;

   typedef double _score_type;
  _score_type score;

   typedef  ::geometry_msgs::PoseWithCovariance_<ContainerAllocator>  _pose_type;
  _pose_type pose;





  typedef boost::shared_ptr< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> const> ConstPtr;

}; // struct ObjectHypothesisWithPose_

typedef ::vision_msgs::ObjectHypothesisWithPose_<std::allocator<void> > ObjectHypothesisWithPose;

typedef boost::shared_ptr< ::vision_msgs::ObjectHypothesisWithPose > ObjectHypothesisWithPosePtr;
typedef boost::shared_ptr< ::vision_msgs::ObjectHypothesisWithPose const> ObjectHypothesisWithPoseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator1> & lhs, const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator2> & rhs)
{
  return lhs.id == rhs.id &&
    lhs.score == rhs.score &&
    lhs.pose == rhs.pose;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator1> & lhs, const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace vision_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
{
  static const char* value()
  {
    return "fa1ab3bc7146f53921fa142d631d02db";
  }

  static const char* value(const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xfa1ab3bc7146f539ULL;
  static const uint64_t static_value2 = 0x21fa142d631d02dbULL;
};

template<class ContainerAllocator>
struct DataType< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
{
  static const char* value()
  {
    return "vision_msgs/ObjectHypothesisWithPose";
  }

  static const char* value(const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# An object hypothesis that contains position information.\n"
"\n"
"# The unique numeric ID of the object class. To get additional information about\n"
"#   this ID, such as its human-readable class name, listeners should perform a\n"
"#   lookup in a metadata database. See vision_msgs/VisionInfo.msg for more detail.\n"
"int64 id\n"
"\n"
"# The probability or confidence value of the detected object. By convention,\n"
"#   this value should lie in the range [0-1].\n"
"float64 score\n"
"\n"
"# The 6D pose of the object hypothesis. This pose should be\n"
"#   defined as the pose of some fixed reference point on the object, such a\n"
"#   the geometric center of the bounding box or the center of mass of the\n"
"#   object.\n"
"# Note that this pose is not stamped; frame information can be defined by\n"
"#   parent messages.\n"
"# Also note that different classes predicted for the same input data may have\n"
"#   different predicted 6D poses.\n"
"geometry_msgs/PoseWithCovariance pose\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/PoseWithCovariance\n"
"# This represents a pose in free space with uncertainty.\n"
"\n"
"Pose pose\n"
"\n"
"# Row-major representation of the 6x6 covariance matrix\n"
"# The orientation parameters use a fixed-axis representation.\n"
"# In order, the parameters are:\n"
"# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)\n"
"float64[36] covariance\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.id);
      stream.next(m.score);
      stream.next(m.pose);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ObjectHypothesisWithPose_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::vision_msgs::ObjectHypothesisWithPose_<ContainerAllocator>& v)
  {
    s << indent << "id: ";
    Printer<int64_t>::stream(s, indent + "  ", v.id);
    s << indent << "score: ";
    Printer<double>::stream(s, indent + "  ", v.score);
    s << indent << "pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::PoseWithCovariance_<ContainerAllocator> >::stream(s, indent + "  ", v.pose);
  }
};

} // namespace message_operations
} // namespace ros

#endif // VISION_MSGS_MESSAGE_OBJECTHYPOTHESISWITHPOSE_H
