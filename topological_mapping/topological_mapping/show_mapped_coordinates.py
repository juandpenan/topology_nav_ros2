import rclpy
import numpy as np
import cv2
import os
import copy
from topological_mapping.topological_mapping.topological_map import TopologicalMap
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from vqa_msgs.msg import VisualFeatures
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from statistics import mean
import pathlib





class ShowResultMap(Node):
    def __init__(self):
        super().__init__('show_map')     

        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability= QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        #parameters
        self.declare_parameter('question_qty', 10) 
        self.declare_parameter('state_qty', 8)
        self.declare_parameter('max_images_per_state', 10)
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.max_images_per_state = self.get_parameter(
                                    'max_images_per_state').get_parameter_value().integer_value       
       
        self.pose = PoseWithCovarianceStamped()     
    
        self.visualizer_id = 0
        self._map_folder = str(pathlib.Path(__file__).parent.resolve()).removesuffix(
                            '/topological_mapping') + '/data/'
        # publishers related variables
        self.marker_publisher = self.create_publisher(MarkerArray, '/mapping_visualizer', 1)
        self.markers_msg = MarkerArray()
        # subscribers related variables
        self.map = None
        self.create_subscription(OccupancyGrid, '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)
      

        self.get_logger().info('show map initialized')

    def visualize_map(self):
        if self.map is None:
            return

        rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix("/topological_mapping") + "/data/"
        for subdir, dirs, files in os.walk(rootdir):
            try:        
                index = int(subdir.removeprefix(rootdir))
                self.get_logger().info('current index: %i' % index)
            except ValueError:
                self.get_logger().error("Fail to convert "+subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/")+" to int")
                continue
            pose = self.map.index_to_pose(index)
            self.plot_index(pose)

    def plot_index(self, pose):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.color.a = 1.0
        msg.scale.x = 0.1
        msg.scale.y = 0.01
        msg.scale.z = 0.01
        msg.type = 0
        msg.pose.position.z = 0.0
        msg.id = self.visualizer_id
        msg.color.r = 1.0
        msg.pose = pose
        self.markers_msg.markers.append(copy.deepcopy(msg))   
        self.marker_publisher.publish(self.markers_msg)
        self.visualizer_id += 1
        return True

      

    def map_callback(self,msg):

        self.map =TopologicalMap(msg,self.state_qty,self.question_qty)
        self.visualize_map()
        


def main(args=None):
    rclpy.init(args=args)

    topo_node = ShowResultMap()

    rclpy.spin(topo_node)


    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

