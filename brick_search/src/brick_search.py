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
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped


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
        self.depth_callback_ready_ = 0
        self.depth_callback_lock_ = Lock()
        self.cam_fov = 100.0
        self.brick_cells_ = [False] * int(self.cam_fov)
        self.brick_coords_ = []
        self.brick_coord_radius_ = 0.01

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

        self.tf_listener_cam_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_cam_.canTransform("map", "camera_link", rospy.Time(0.)):
            rospy.sleep(0.1)

        # Advertise brick markers to RViZ
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)

        # Subscribe to depth camera
        self.depth_sub_ = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)

        # Subscribe to the camera
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, callback_args='image_msg', queue_size=1)

        # Advertise "cmd_vel" publisher to control TurtleBot manually
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base action available")

        # Reinitialise AMCL
        global_localization_service_client = rospy.ServiceProxy('global_localization', Empty)
        empty = global_localization_service_client()
        rospy.sleep(0.5)

        # Subscribe to "amcl_pose" to get pose covariance
        self.amcl_pose_sub_ = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_callback, queue_size=1)


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

    def get_pose_2d_cam(self):

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'camera_link', rospy.Time(0))

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


    def image_callback(self, image_msg, depth_msg):
        # Use this method to identify when the brick is visible

        # The camera publishes at 30 fps, it's probably a good idea to analyse images at a lower rate than that
        if self.image_msg_count_ < 15:
            self.image_msg_count_ += 1
            return
        else:
            self.image_msg_count_ = 0

            if self.brick_found_:
                with self.depth_callback_lock_:
                    self.depth_callback_ready_ = 1

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

        if not self.brick_found_:
            self.brick_cells_ = [False] * int(self.cam_fov)

        with self.depth_callback_lock_:
            for index in range(int(self.cam_fov)):
                # Divide the image into 100 segments (one segment per one degree of FOV)
                left = index*int(width*(1.0/self.cam_fov))
                right = (index + 1)*int(width*(1.0/self.cam_fov))
                segment = image[ : , left : right]

                # Designate if red range is in the segment and assign answer to list
                has_brick = cv.countNonZero(cv.inRange(segment, lower_range_red, upper_range_red)) > 0

                # Assign condition
                self.brick_cells_[index] = has_brick
            
            # Remove potentially partially filled segments (first and last)
            # e.g., some brick is captured some environment is also captured.
            # This ensures Depth Cam measurement touches the brick and not the environment.
            
            try:
                # Clear first and last FOV elements where brick is visible
                # (Must be visible across four degrees minimum to yield a marker placement)
                for _ in range(2):
                    self.brick_cells_[self.brick_cells_.index(True)] = False
                    self.brick_cells_.reverse()
            except ValueError:
                self.brick_cells_ = [False] * int(self.cam_fov)

        # You can set "brick_found_" to true to signal to "mainLoop" that you have found a brick
        # You may want to communicate more information
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "brick_found_" doesn't need a mutex because it's an atomic




        rospy.loginfo('image_callback')
        rospy.loginfo('brick_found_: ' + str(self.brick_found_))

    def depth_callback(self, depth_msg):
        if not self.brick_found_:
            return
        # Synchronise Depth Cam callback with RGB Cam callback
        with self.depth_callback_lock_:
            if self.depth_callback_ready_:
                self.depth_callback_ready_ = 0
                
                # if brick is not seen end the method early
                brick_seen = False
                for index in range(len(self.brick_cells_)):
                    if self.brick_cells_[index]:
                        brick_seen = True
                        break
                if not brick_seen:
                    return

                # Make local copy of brick_cells
                brick_cells_ = self.brick_cells_
            else:
                return

        rospy.loginfo("depth_callback")

        depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg)
        # Get cam distances
        height, width = depth.shape
        depth = depth[int(height*0.5):int(height*0.5)+1, : ] # Slice depth image to single list (Single pixel row)
        depth = depth[0]

        # Keep a list of distances whose depth cam rays touch Wally
        valid_distances = []

        quotient = int(width/self.cam_fov)
        for index in range(int(self.cam_fov)):
            if brick_cells_[index]:
                if depth[quotient*index] == np.nan:
                    rospy.loginfo("Wally is too close!") # Caveat of the depth cam
                    return
                valid_distances.append((index, depth[quotient*index]))
        
        if len(valid_distances) == 0:
            return
        
        cam_pose = self.get_pose_2d_cam()

        coordinates = []

        for index in range(len(valid_distances)):
            coordinates.append(self.get_coordinate(cam_pose, valid_distances[index][0], valid_distances[index][1]))

        self.publish_brick_marker(coordinates)


    # Given the current robot pose, cam index (angle) and distance value from Depth Cam get the world coordinate of Depth Cam distance ray hitting an object
    def get_coordinate(self, pose, angle_offset, dist):
        angle = wrap_angle(pose.theta + math.radians(self.cam_fov/2.0) - math.radians(angle_offset))
        
        x = pose.x + dist*math.cos(angle)
        y = pose.y + dist*math.sin(angle)

        return (x, y)
    
    # Maintains a set list of spaced world coordinates to create a rectangular marker for the brick
    def publish_brick_marker(self, coordinates):
        # Update point list
        old_length_ = len(self.brick_coords_)
        for index in range(len(coordinates)):
            if len(self.brick_coords_) == 0:
                self.brick_coords_.append(coordinates[index])
            else:
                skip = False
                for existing_coordinate in self.brick_coords_:
                    pair = [coordinates[index], existing_coordinate]
                    # Set new coordinates outside of radius to minimize cam measurement jitter
                    if self.get_distance(pair) < self.brick_coord_radius_:
                        skip = True
                        break
                if not skip:
                    self.brick_coords_.append(coordinates[index])
        
        # Don't try to publish if no new coordinates added or too few points to make a marker
        if old_length_ == len(self.brick_coords_) or len(self.brick_coords_) <= 1:
            return
        
        # Gets either the straight line (one face of the brick) or the coordinates of the hypotenuse/diagonal of the brick
        best_pair = self.get_max_distance_coordinates(self.brick_coords_)

        # Finds the midpoint of the diagonal a.k.a. the centroid
        x, y = self.get_midpoint_coordinate(best_pair)

        vertex = self.check_for_vertices(best_pair, self.brick_coords_)

        rotation = self.get_angle(best_pair, vertex)

        x_scale, y_scale = self.get_axis_scales(best_pair, vertex)

        marker = self.create_marker(x, y, x_scale, y_scale, rotation)

        self.marker_pub.publish(marker)

        rospy.loginfo(f"Placed marker at world point: {(x, y)}")

    # Returns the world midpoint of two points
    def get_midpoint_coordinate(self, coordinates):
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]

        return ((x1 + x2)/2, (y1 + y2)/2)

    # Find the pair of coordinates in a list whose distance between them is furthest apart
    def get_max_distance_coordinates(self, coordinates):
        max_distance = 0.0
        coordinate_pair = []
        for index_A in range(len(coordinates)):
            for index_B in range(len(coordinates)):
                pair = [coordinates[index_A], coordinates[index_B]]
                distance = self.get_distance(pair)
                if distance > max_distance:
                    max_distance = distance
                    coordinate_pair = pair
        return coordinate_pair
        
    def get_distance(self, coordinates):
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]

        return math.sqrt((x2-x1)**2.0 + (y2-y1)**2.0)
    
    # Gets marker scales from distance of straight line or distance from diagonal points to corner point
    def get_axis_scales(self, coordinates, vertex):
        if vertex is None:
            # Can be thicker or thinner due to cam jitter
            x_delta = coordinates[0][0] - coordinates[1][0]
            y_delta = coordinates[0][1] - coordinates[1][1]
        else:
            if self.get_distance((coordinates[0], vertex)) > self.get_distance((coordinates[1], vertex)):
                x_delta = self.get_distance((coordinates[0], vertex))
                y_delta = self.get_distance((coordinates[1], vertex))
            else:
                x_delta = self.get_distance((coordinates[1], vertex))
                y_delta = self.get_distance((coordinates[0], vertex))

        return (x_delta, y_delta)

    # Gets the angle of rotation of the brick or straight line if no corner is present.
    def get_angle(self, pair, vertex):
        if vertex is not None:
            if self.get_distance((pair[0], vertex)) > self.get_distance((pair[1], vertex)):
                x_delta = pair[0][0] - vertex[0]
                y_delta = pair[0][1] - vertex[1]
            else:
                x_delta = pair[1][0] - vertex[0]
                y_delta = pair[1][1] - vertex[1]
        else:
            x_delta = pair[0][0] - pair[1][0]
            y_delta = pair[0][1] - pair[1][1]

        return math.atan2(y_delta, x_delta)
    
    # Tries to find if a point is a corner, given two ends of a line and another point:
    # the internal angle becomes very close to 90 degrees i.e., oppposite the hypotenuse.
    def check_for_vertices(self, pair, coordinates):
        corner_point = None
        best_angle = 0

        for coordinate in coordinates:
            # Don't use points that are the same as the current coordinates of the diagonal
            if pair[0] == coordinate or pair[1] == coordinate:
                continue

            line_A = self.get_distance((pair[0], coordinate))
            line_B = self.get_distance((pair[1], coordinate))
            line_C = self.get_distance((pair[0], pair[1]))

            angle = math.degrees(math.acos((line_A**2 + line_B**2 - line_C**2)/(2*line_A*line_B)))
            
            # Find angles relatively close to 90 degrees
            if angle > 85.0 and angle < 115.0:
                # Set corner point whose angle is closest to 90 degrees
                if abs(angle - 90) < abs(best_angle - 90):
                    best_angle = angle
                    corner_point = coordinate
                
        return corner_point
    
    # Creates a basic 2D red marker
    def create_marker(self, x, y, x_scale, y_scale, rot):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "wally_brick"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = x_scale
        marker.scale.y = y_scale
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        # Quaternion Rotation (Rotation around the Z-Axis to line up the marker)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(rot/2.0)
        marker.pose.orientation.w = math.cos(rot/2.0)

        return marker

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




