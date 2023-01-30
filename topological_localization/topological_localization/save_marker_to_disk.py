import pickle
from visualization_msgs.msg import Marker, MarkerArray
import copy
import os
import pathlib
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from geometry_msgs.msg import Point
from topological_mapping.topological_mapping.topological_map import TopologicalMap
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy


class SaveMsgToDisc(Node):

    def __init__(self):
        super().__init__('msg_array_Saver')

        map_qos_profile = QoSProfile(
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        history=QoSHistoryPolicy.KEEP_LAST,
                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                        depth=1)
        #parameters
        self.declare_parameter('question_qty', 10)
        self.declare_parameter('state_qty', 8)
        self.declare_parameter('max_images_per_state', 10)

        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.question_depth = self.get_parameter('max_images_per_state').get_parameter_value().integer_value

        self.msg_path = str(pathlib.Path(__file__).parent.resolve()).removesuffix('/topological_localization') + '/resource/msg.pkl'

        duration = Duration()
        duration.sec = 1
        self.marker_array_msg = MarkerArray()
        self.marker_msg = Marker()
        self.marker_msg.header.frame_id = 'map'
        self.marker_msg.color.r = 1.0
        self.marker_msg.scale.x = 0.05
        self.marker_msg.scale.y = 0.05
        self.marker_msg.scale.z = 0.002
        # self.marker_msg.scale.x = 1.0
        # self.marker_msg.scale.y = 1.0
        # self.marker_msg.scale.z = 1.0
        self.marker_msg.type = 6
        self.marker_msg.pose.position.z = 0.0
        # self.marker_msg.lifetime = copy.deepcopy(duration)
        self.id = 0

        self.marker_publisher = self.create_publisher(Marker, '/localization_visualizer', 1)
        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)

    def map_callback(self, map_msg):

        self.map_helper = TopologicalMap(map_msg,
                                         self.state_qty,
                                         self.question_qty,
                                         self.question_depth)
        self._localization_grid = np.full(
            shape=(self.map_helper.occupancy_map.info.height,
                   self.map_helper.occupancy_map.info.width,
                   self.state_qty+1),
            fill_value=1/((self.state_qty+1)*self.map_helper.occupancy_map.info.height *
                          self.map_helper.occupancy_map.info.width))

        self.save_to_disc(self._localization_grid)

    def save_to_disc(self, grid):
        non_zero_indices = np.transpose(np.nonzero(grid[:, :, 1:]))
        points = []
        point = Point()
        for row, col, state in non_zero_indices:

            self.marker_msg.id = self.id
            self.marker_msg.color.a = 1.0
            self.id += 1

            x, y = self.map_helper._get_world_x_y(col, row)
            # angle = self.map_helper._undiscretize_angle(state)

            self.marker_msg.pose.position.x = 0.0
            self.marker_msg.pose.position.y = 0.0

            q = self.map_helper._quaternion_from_euler(0, 0, 0.0)
            
            self.marker_msg.pose.orientation.x = q[0]
            self.marker_msg.pose.orientation.y = q[1]
            self.marker_msg.pose.orientation.z = q[2]
            self.marker_msg.pose.orientation.w = q[3]

            point.x = x
            point.y = y
            point.z = 0.0

            points.append(point)
            
            self.marker_msg.points = points

            self.marker_publisher.publish(self.marker_msg)

        # self.marker_array_msg.markers.append(copy.deepcopy(self.marker_msg))
        # while True:
        #     
            

            # with open(self.msg_path, 'wb') as outp:
            #     pickle.dump(self.marker_array_msg, outp, pickle.HIGHEST_PROTOCOL)
        
        # self.get_logger().info('saved, shutting down .. ')
        # self.destroy_node()
    
        return True


def main(args=None):

    rclpy.init(args=args)

    node = SaveMsgToDisc()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()