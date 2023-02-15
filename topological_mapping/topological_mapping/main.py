import copy
import os
import pathlib

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy

from message_filters import ApproximateTimeSynchronizer, Subscriber

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from vqa_msgs.msg import VisualFeatures

from topological_mapping.topological_mapping.topological_map import TopologicalMap

from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class TopologicalMapping(Node):
    def __init__(self):
        super().__init__('topological_mapping')     

        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        # parameters
        self.declare_parameter('question_qty', 10)
        self.declare_parameter('state_qty', 8)
        self.declare_parameter('max_images_per_state', 10)
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.max_images_per_state = self.get_parameter(
                                    'max_images_per_state').get_parameter_value().integer_value
        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.previous_state = None
        self.state_counter = 0
        self.features = VisualFeatures()
        self.pose = PoseWithCovarianceStamped()
        self._update_counter = 0
        self.image_converter = CvBridge()
        self.first_run = True
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
        self.tss = ApproximateTimeSynchronizer([Subscriber(self, Image, 'image'),
                                                Subscriber(self, Odometry, 'odom')],
                                               10,
                                               0.25)
        self.tss.registerCallback(self.store_data_cb)

        self.broadcast_map()
        self.get_logger().info('topological_mapping initialized')

    def broadcast_map(self):

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t)

    def store_data_cb(self, image_msg, odom_msg):
        try:
            self.map.state_qty
        except Exception:
            self.get_logger().warning('map has not been initialized yet')
            return
        else:
            state = str(self.map.pose_to_index(pose=odom_msg))
            self.get_logger().info(f'current state: {state}')
            if self._check_saved_images(state) < (self.max_images_per_state):
                self.__save_image_to_disc(image_msg, odom_msg)
                self.visualize_mapping(odom_msg.pose.pose)
            else:
                self.get_logger().info('this state already reached limit images')
        return True

    def _check_saved_images(self, state):
        if not os.path.exists(self._map_folder + state):
            self.get_logger().info('STATE NOT CREATED')
            file_count = 0
        else:
            _, _, files = next(os.walk(os.path.join(self._map_folder + state)))
            file_count = len(files)
        self.get_logger().debug(f'file count {file_count}')
        return int(file_count)

    def __save_image_to_disc(self, img, pose):
        folder_name = str(self.map.pose_to_index(pose=pose))
        _path = os.path.join(self._map_folder + folder_name)
        os.makedirs(_path, exist_ok=True)
        image = self.image_converter.imgmsg_to_cv2(img)      	
        # cv2.imwrite(_path+"/"+str(self.get_clock().now().nanoseconds)+".jpg",  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(_path+'/'+str(self.get_clock().now().nanoseconds)+'.jpg',  image)

    def visualize_mapping(self, pose):
        msg = Marker()
        msg.header.frame_id = 'map'
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

    def map_callback(self, msg):

        self.map = TopologicalMap(msg, self.state_qty, self.question_qty)


def main(args=None):
    rclpy.init(args=args)

    topo_node = TopologicalMapping()

    rclpy.spin(topo_node)

    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
