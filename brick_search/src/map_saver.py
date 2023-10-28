#!/usr/bin/env python3

import rospy
import subprocess
from move_base_msgs.msg import MoveBaseActionResult

class MapSaver:

    def __init__(self):
        # Initialize node
        rospy.init_node('auto_map_saver')
        
        # Subscribe to the result topic of move_base to check if the goal was reached or aborted
        rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.callback)
        
        # Parameter to check if map has been saved already (to prevent multiple saves)
        self.map_saved = False

    def callback(self, data):
        # Check if the goal was aborted, which might indicate exploration completion
        if data.status.status == 4 and not self.map_saved:
            rospy.loginfo("Exploration might be complete. Saving map...")
            
            # Call map_saver utility
            try:
                subprocess.check_call(["rosrun", "map_server", "map_saver", "-f", "/home/emmy/catkin_ws/src/brick_search/config"])
                self.map_saved = True
            except subprocess.CalledProcessError:
                rospy.logerr("Failed to save the map.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = MapSaver()
    node.run()

