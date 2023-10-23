import rclpy
import numpy as np
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from nav2_msgs.msg import Costmap
from map_msgs.srv import GetMapROI
from std_msgs.msg import Header, Float32
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point
from std_msgs.msg import Float32MultiArray
from vqa_msgs.msg import VisualFeatures, MonologueHypothesis
from vqa_msgs.srv import Hypothesis
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
import pathlib
from transformers import RobertaModel,RobertaTokenizerFast
from scipy import ndimage, signal
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import math
import PyKDL
import tf2_kdl
from visualization_msgs.msg import Marker, MarkerArray
from message_filters import ApproximateTimeSynchronizer, Subscriber, Cache
# import tf_transformations
from topological_mapping.topological_mapping.topological_map import TopologicalMap
from topological_localization.visualizer import Visualizer
from geometry_msgs.msg import Transform, TransformStamped



class TopologicalLocalization(Node):
    def __init__(self):
        super().__init__('topological_localization')

        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # parameters
        self.declare_parameter('map_resolution', 0.050)
        # self.declare_parameter('kernel_scale', 2.0)
        self.declare_parameter('question_qty', 10)
        self.declare_parameter('state_qty', 8)
        self.declare_parameter('max_images_per_state', 10)
        self.declare_parameter('map_name', "map")

        
        # it is the number the map(gridmap) shape will be divided by
        # self.get_parameter('kernel_scale').get_parameter_value().double_value
        self.kernel_scale = 8
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.question_depth = self.get_parameter(
                             'max_images_per_state').get_parameter_value().integer_value
        # m/pix
        self.map_resolution = self.get_parameter(
                             'map_resolution').get_parameter_value().double_value
        self.map_name = self.get_parameter(
            'map_name').get_parameter_value().string_value
       

        self.__pkg_folder = str(
                            pathlib.Path(__file__).parent.resolve()).removesuffix(
                                '/topological_localization')

        self.map_folder = os.path.join(get_package_share_directory('topological_mapping'),
                                       self.map_name + '.npy')
        self.image_map_folder = os.path.join(get_package_share_directory('topological_mapping'),
                                             self.map_name + '.jpg')

        self.image_converter = CvBridge()
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.map_helper = None
        # init prediction variables
        self.vqa_features = None
        self.control_cycle_timer = self.create_timer(1.5, self.control_cycle)
        self.reset_grid = self.create_timer(3.0, self.reset_grid)

        # services 
        self.create_service(Hypothesis, 'get_hypothesis', self.get_hypothesis)

        # publishers
        self.pose_publisher = self.create_publisher(MonologueHypothesis, '/markov_pose', 1)

        self.pose_publisher_variant = self.create_publisher(
            MarkerArray, '/markov_pose_variant', 1)

        
        # self.pose_publisher_variant = self.create_publisher(
        #     MarkerArray, '/markov_pose_variant', 1)

        self.grid_publisher = self.create_publisher(
            OccupancyGrid, '/motion_update/localization_grid', 1)
        # subscribers

        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)
        # remapped from your odom topic(ground_truth_odom) to /odom

        self.create_subscription(VisualFeatures, '/vqa/features', self.vqa_callback, 1)

        self.broadcast_map()

    def get_hypothesis(self, request, response):
        if self.vqa_features is None or self.map_helper is None:
            return response
        
        self.reset_grid()
        self.localization_algorithm()
        t = Transform()

        ind = np.unravel_index(np.argmax(self._localization_grid, axis=None),  self._localization_grid.shape)

        x, y = self.map_helper._get_world_x_y(ind[-1], ind[0])

        t.translation.x = x
        t.translation.y = y
        t.orientation.w = 1.0

        response.hypothesis = t



        return response

    def reset_grid(self):
        if self.vqa_features is None or self.map_helper is None:
            return

        self.init_localization_grid()

    def control_cycle(self):

        if self.vqa_features is None or self.map_helper is None:
            return

        self.get_logger().info('executing algorithm')

        self.localization_algorithm()
        image = self.grid_to_img()
        map = self.img_to_occupancy(image)
        self.grid_publisher.publish(map)
        # self.init_localization_grid()



    def map_callback(self, msg):

        self.map_helper = TopologicalMap(msg,
                                         self.state_qty,
                                         self.question_qty,
                                         self.question_depth)

        self.get_logger().info('loading topological map .. ')
        self.get_logger().info('map folder path  ' + self.map_folder)
        self.map_helper.load_map(self.map_folder)
        self.init_localization_grid()


    def grid_to_img(self):

        return (self._localization_grid * 255).round().astype(np.uint8)

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

    def perception_update(self):

        question_answers_indexes = []
        question_answers_accs = []
        
        for i in range(len(self.vqa_features.data)):
            # the nos seems to dont bring more information (to check)
            if self.vqa_features.data[i] == 'no':
                continue
            ind = np.where(self.map_helper.topological_map['q_a'] == self.vqa_features.data[i])
            # we keep only coincidences in the current question
            ind = ind[0][np.where(ind[1] == i)]
            
            # we keep the topological indexes where there is a coincidence :
            current_question_indexes = self.map_helper.topological_map['index'][np.unique(ind)]
            current_question_acc = []
 
            # we extract the accuracy for each one of them (acc of question times acc of map)            
            for index in np.unique(ind):
                # model 1 :
                acc = 1/(abs(self.vqa_features.acc[i] - np.mean(self.map_helper.topological_map['acc'][index][i][np.nonzero(self.map_helper.topological_map['acc'][index][i])]))+0.2)

                # model 2 : 
                # acc_ind = np.where(self.map_helper.topological_map['q_a'][index][i] == self.vqa_features.data[i])                        
                # acc = acc_ind[0].size / np.nonzero(self.map_helper.topological_map['q_a'][index][i])[0].size
                
                # model 3:
                # acc = np.sum(self.map_helper.topological_map['acc'][index][i][np.nonzero(self.map_helper.topological_map['acc'][index][i])]) / self.map_helper.topological_map['acc'][index][i][np.nonzero(self.map_helper.topological_map['acc'][index][i])].size
                
                # test model:
                # acc = 1.0
                current_question_acc.append(acc)

            question_answers_indexes.extend(current_question_indexes.tolist())
            question_answers_accs.extend(current_question_acc)

                    
            current_map_raw = np.transpose(np.array([question_answers_indexes, question_answers_accs]))
      
            
            # there are repeated indexes
            unique_elements, counts = np.unique(current_map_raw[:, 0], return_counts=True)
            # self.get_logger().info(f'current map {current_map_raw[:, 0]}')

        
        #     # current_map_raw[index][1]
        #     # Iterate over the unique elements
      
            for x, i  in enumerate(unique_elements):
                idx = np.where(current_map_raw[:, 0] == i)
                col,row,state = self.map_helper.topological_index_to_occupancy_x_y(int(i))

                # # model 1 
                acc = np.sum(current_map_raw[idx, 1])


                # model 3:
                # acc = current_map_raw[idx, 1]

                # model 1
                self._localization_grid[row,col] *= acc
                self._localization_grid_yaw[state] *= acc

                # model 2 :
                # self._localization_grid[row,col] += counts[x] / (self.question_qty - 0)
                # self._localization_grid_yaw[state] += counts[x] / (self.question_qty - 0)

                # model 3:
                # self._localization_grid[row,col] *= np.sum(acc) / (self.question_qty - 2)
                # self._localization_grid_yaw[state] *= np.sum(acc) / (self.question_qty - 2)

                # model test:
                # self._localization_grid[row,col] += 1 

        self._localization_grid = self._localization_grid / np.sum(self._localization_grid)
        self._localization_grid = self._localization_grid / self._localization_grid.max()

        self._localization_grid_yaw = self._localization_grid_yaw / np.sum(self._localization_grid_yaw)
        self._localization_grid_yaw = self._localization_grid_yaw / self._localization_grid_yaw.max()






    def img_to_occupancy(self,image):

        if self.map_helper == None:
            return
        
        image = image * 100
        map = self.map_helper.occupancy_map
        data = np.array(image.flatten(), dtype=np.int8)
        map.data = data.tolist()

        return map


    def vqa_callback(self,vqa_msg):
    
        self.vqa_features = vqa_msg




    def init_localization_grid(self):


        self._localization_grid = np.full(
            shape=(self.map_helper.occupancy_map.info.height,
                   self.map_helper.occupancy_map.info.width,                   
                   ),
            fill_value= 1/ (self.map_helper.occupancy_map.info.height *
                   self.map_helper.occupancy_map.info.width ))
        
        self._localization_grid_yaw = np.full(shape=(self.state_qty,), fill_value=1/self.state_qty)

        self._localization_grid = self._localization_grid / np.sum(self._localization_grid)
        self._localization_grid = self._localization_grid / self._localization_grid.max()

        self._localization_grid_yaw = self._localization_grid_yaw / np.sum(self._localization_grid_yaw)
        self._localization_grid_yaw = self._localization_grid_yaw / self._localization_grid_yaw.max()


        self.get_logger().info("grid initialized")

 
        return True

   
    def _calculate_sigma(self):
        pass
    

    def publish_pose(self):
        if self.vqa_features is None :
             return
        
        msg = MonologueHypothesis()

        indexes = np.argsort(-self._localization_grid.flatten())
      
        top3_indexes_2d = np.unravel_index(indexes[:4], self._localization_grid.shape)

        angle_indexes = np.argsort(-self._localization_grid_yaw.flatten())
        top_three_angle_indices = angle_indexes[:4]
        marker_array = MarkerArray()
        i = 0
        header = Header()
        header.frame_id = 'map'

        poses = []
        weights = []
        headers = []
        for ind_y, ind_x in zip(*top3_indexes_2d):

            pose = Pose()
            


            x, y = self.map_helper._get_world_x_y(ind_x, ind_y)
                    
            theta = self.map_helper._undiscretize_angle(angle_indexes[i])
            q = self.map_helper._quaternion_from_euler(0.0, 0.0, theta) 
            
            pose.position.x = x 
            pose.position.y = y
            pose.position.z = 0.0

            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            poses.append(pose)
            weights.append(0.51)
            headers.append(header)


            i += 1
        

        msg.poses = poses
        msg.weight = weights
        msg.headers = headers

        self.pose_publisher.publish(msg)

        return True
        
    def draw_poses(self):

        indexes = np.argsort(-self._localization_grid.flatten())
      
        top3_indexes_2d = np.unravel_index(indexes[:4], self._localization_grid.shape)

        angle_indexes = np.argsort(-self._localization_grid_yaw.flatten())
        top_three_angle_indices = angle_indexes[:4]
        marker_array = MarkerArray()
        i = 0
        for ind_y, ind_x in zip(*top3_indexes_2d):

            marker = Marker()
            
            header = Header()
            header.frame_id = "map"
            header.stamp = self.get_clock().now().to_msg()

            marker.header = header
            marker.id = i

            marker.action = 0
            marker.scale.x = 0.5
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            if i == 0:
                marker.color.a = 1.0
            elif i == 1 :
                marker.color.a = 0.8
            elif i == 2 :
                marker.color.a = 0.6
            elif i == 3 :
                marker.color.a = 0.4
            else:
                marker.color.a = 1.0
            marker.color.r = 1.0

            x, y = self.map_helper._get_world_x_y(ind_x, ind_y)
                    
            theta = self.map_helper._undiscretize_angle(angle_indexes[i])
            q = self.map_helper._quaternion_from_euler(0.0, 0.0, theta) 
            
            marker.pose.position.x = x 
            marker.pose.position.y = y
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker_array.markers.append(marker)
            i += 1 

        self.pose_publisher_variant.publish(marker_array)

 
    def localization_algorithm(self):
        
        t1 = self.get_clock().now()
        self.perception_update()
        t2 = self.get_clock().now()
        self.draw_poses()
        self.publish_pose()
        self.get_logger().info(f'time on perception update :{t2-t1}')

   
 




def main(args=None):
    rclpy.init(args=args)

    topo_node = TopologicalLocalization()

    rclpy.spin(topo_node)


    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

