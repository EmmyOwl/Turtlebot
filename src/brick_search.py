#!/usr/bin/env python3

import rospy
import roslib
import math
import cv2 as cv # OpenCV2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.srv import GetMap
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan




def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose

class BrickSearch:
    def __init__(self):

        # Variables/Flags
        self.localised_ = False
        self.brick_found_ = False
        self.image_msg_count_ = 0

        # Get the map via a ROS service call
        rospy.loginfo("Waiting for static_map service...")
        rospy.wait_for_service('static_map')
        get_map_service = rospy.ServiceProxy('static_map', GetMap)
        try:
            resp = get_map_service()
            self.map_ = resp.map
        except rospy.ServiceException as exc:
            rospy.logerror('Service did not process request: ' + str(exc))
            rospy.signal_shutdown('Service did not process request')
        rospy.loginfo("Map received")

        # Convert map into a CV image
        self.cv_bridge_ = CvBridge()
        self.map_image_ = np.reshape(self.map_.data, (self.map_.info.height, self.map_.info.width)).astype(np.int32)

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)

        # Subscribe to the camera
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

        # Advertise "cmd_vel" publisher to control TurtleBot manually
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base action available")

        # Reinitialise AMCL
        global_localization_service_client = rospy.ServiceProxy('global_localization', Empty)
        #empty = global_localization_service_client()
        rospy.sleep(0.5)

        # Subscribe to "amcl_pose" to get pose covariance
        self.amcl_pose_sub_ = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_callback, queue_size=1)

        self.marker_pub_ = rospy.Publisher('brick_marker', Marker, queue_size=10)

        self.laser_sub_ = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)
        self.latest_scan_ = None

    def laser_callback(self, data):
        self.latest_scan_ = data

    def get_distance_in_front(self):
        # Assuming laser scanner is front-facing and has a 180-degree field of view
        # If the robot is oriented such that 0-degree is front, then:
        if self.latest_scan_ is None:
            return None  # No data received yet

        front_distance = self.latest_scan_.ranges[len(self.latest_scan_.ranges) // 2]
        return front_distance if front_distance < self.latest_scan_.range_max else None

    def get_pose_2d(self):

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

        print(trans)
        print(rot)

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]

        qw = rot[3];
        qz = rot[2];

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw));

        return pose

    def amcl_pose_callback(self, pose_msg):

        # Check the covariance
        frobenius_norm = 0.0

        for e in pose_msg.pose.covariance:
            frobenius_norm += e**2

        if frobenius_norm < 0.05:
            self.localised_ = True

            # Unsubscribe from "amcl_pose" because we should only need to localise once at start up
            self.amcl_pose_sub_.unregister()


    def image_callback(self, image_msg):
        # Use this method to identify when the brick is visible

        print("start image_callback")

        # The camera publishes at 30 fps, it's probably a good idea to analyse images at a lower rate than that
        if self.image_msg_count_ < 15:
            self.image_msg_count_ += 1
            return
        else:
            self.image_msg_count_ = 0

        # Copy the image message to a cv_bridge image
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg)

        # Convert to RGB format
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Reduce image height to 2%
        height, width, _ = image.shape

        top = int(height*0.49)
        bottom = int(height*0.51)

        image = image[top:bottom, :] # Slice image

        # Specify 'Red' Colour Range
        lower_range_red = np.array([0, 0, 100], dtype=np.uint8)
        upper_range_red = np.array([40, 40, 255], dtype=np.uint8)

        # Create a bit mask that specifies pixels in the correct colour range
        bit_mask = cv.inRange(image, lower_range_red, upper_range_red)

        red_present = cv.countNonZero(bit_mask) > 0

        self.brick_found_ = red_present
        
        # You can set "brick_found_" to true to signal to "mainLoop" that you have found a brick
        # You may want to communicate more information
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "brick_found_" doesn't need a mutex because it's an atomic

        rospy.loginfo('image_callback')
        rospy.loginfo('brick_found_: ' + str(self.brick_found_))

        if self.brick_found_:
            distance_to_brick = self.get_distance_in_front()
            if distance_to_brick is not None:
                # Get the robot's current pose
                pose_2d = self.get_pose_2d()

                # Calculate the position of the brick relative to the robot's current position
                brick_position = Pose2D()
                brick_position.x = pose_2d.x + distance_to_brick * math.cos(pose_2d.theta)
                brick_position.y = pose_2d.y + distance_to_brick * math.sin(pose_2d.theta)
                brick_position.theta = pose_2d.theta

                # Create a marker for the brick's position
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "brick_marker"
                marker.id = 0
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = brick_position.x
                marker.pose.position.y = brick_position.y - 0.065
                marker.pose.position.z = 0.094
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = 0.3  
                marker.scale.y = 0.3
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                # Publish the marker
                self.marker_pub_.publish(marker)

                rospy.loginfo('Brick_Marker Set')


    def main_loop(self):

        # Wait for the TurtleBot to localise
        rospy.loginfo('Localising...')
        while not rospy.is_shutdown():

            # Turn slowly
            twist = Twist()
            twist.angular.z = 1.
            self.cmd_vel_pub_.publish(twist)

            if self.localised_:
                rospy.loginfo('Localised')
                break

            rospy.sleep(0.1)

        # Stop turning
        twist = Twist()
        twist.angular.z = 0.
        self.cmd_vel_pub_.publish(twist)

        # The map is stored in "map_"
        # You will probably need the data stored in "map_.info"
        # You can also access the map data as an OpenCV image with "map_image_"

        # Here's an example of getting the current pose and sending a goal to "move_base":
        pose_2d = self.get_pose_2d()

        rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

        # Move forward 0.5 m
        pose_2d.x += 0.5 * math.cos(pose_2d.theta)
        pose_2d.y += 0.5 * math.sin(pose_2d.theta)

        rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

        rospy.loginfo('Sending goal...')
        self.move_base_action_client_.send_goal(action_goal.goal)

        # This loop repeats until ROS is shutdown
        # You probably want to put all your code in here
        while not rospy.is_shutdown():

            rospy.loginfo('main_loop')

            # Get the state of the goal
            state = self.move_base_action_client_.get_state()

            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())

            if state == actionlib.GoalStatus.SUCCEEDED:

                rospy.loginfo('Action succeeded!')

                # Shutdown when done
                rospy.signal_shutdown('Action succeeded!')

            # Delay so the loop doesn't run too fast
            rospy.sleep(0.2)


if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('brick_search')

    # Create the brick search
    brick_search = BrickSearch()

    # Loop forever while processing callbacks
    brick_search.main_loop()




