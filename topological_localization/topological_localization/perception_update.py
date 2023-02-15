import rclpy
import numpy as np
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from nav2_msgs.msg import Costmap
from map_msgs.srv import GetMapROI
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from vqa_msgs.msg import VisualFeatures
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
from geometry_msgs.msg import TransformStamped



class TopologicalLocalization(Node):
    def __init__(self):
        super().__init__('topological_localization')

        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability= QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # parameters
        self.declare_parameter('map_resolution', 0.050)
        # self.declare_parameter('kernel_scale', 2.0)
        self.declare_parameter('question_qty', 10)
        self.declare_parameter('state_qty', 8)
        self.declare_parameter('max_images_per_state', 10)
        
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

        self.__pkg_folder = str(
                            pathlib.Path(__file__).parent.resolve()).removesuffix(
                                '/topological_localization')

        self.map_folder = os.path.join(
            get_package_share_directory('topological_mapping'), 'map6.npy')
        self.image_map_folder = os.path.join(
            get_package_share_directory('topological_mapping'), 'map6.jpg')

        self.image_converter = CvBridge()
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.map_helper = None
        # init prediction variables
        self.vqa_features = None
        self.timer = self.create_timer(7, self.control_cycle)

        # publishers
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/markov_pose', 1)
        self.pose_publisher_variant = self.create_publisher(
            PoseWithCovarianceStamped, '/markov_pose_variant', 1)

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

    def control_cycle(self):

        if self.vqa_features == None:
            return

        self.get_logger().info('executing algorithm')

        self.localization_algorithm()
        # image = self.grid_to_img()
        # map = self.img_to_occupancy(image)

        # self.grid_publisher.publish(map)
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
        return (self._localization_grid[:,:,0] * 255).round().astype(np.uint8)

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

        # ['garage'],
        # ['toaster oven', 'printer'],
        # ['washing machine', 'pillow'],
        # ['plastic', 'plastic'],
        # ['square', 'square']
        # test = ['garage', 'toaster oven', 'washing machine', 'plastic', 'square']
        test = ['garage', 'toaster oven', 'pillow', 'plastic', 'square']
        self.vqa_features.data = self.vqa_features.data[0:3]
        # self.vqa_features.data = self.vqa_features.data[:-1]
        question_answers_indexes = []
        question_answers_accs = []
        self.get_logger().debug(f'must be 5 {len(self.vqa_features.data)}')
        for i in range(len(self.vqa_features.data)):


            ind = np.where(self.map_helper.topological_map['q_a'] == self.vqa_features.data[i])
            # we keep only coincidences in the current question
            ind = ind[0][np.where(ind[1] == i)]
            
            # we keep the topological indexes where there is a coincidence :
            current_question_indexes = self.map_helper.topological_map['index'][np.unique(ind)] 
            current_question_acc = []
 
            # we extract the accuracy for each one of them (acc of question times acc of map)            
            for index in np.unique(ind):
                acc_ind = np.where(self.map_helper.topological_map['q_a'][index][i] == self.vqa_features.data[i])            
                acc = acc_ind[0].size / np.nonzero(self.map_helper.topological_map['q_a'][index][i])[0].size              
                current_question_acc.append(acc)

            question_answers_indexes.extend(current_question_indexes.tolist())
            question_answers_accs.extend(current_question_acc)

                    
            current_map_raw = np.transpose(np.array([question_answers_indexes,question_answers_accs]))
            
            # there are repeated indexes
            unique_elements, counts = np.unique(current_map_raw[:, 0], return_counts=True)

         


            # Iterate over the unique elements
            x=0
            for i in unique_elements:

                col,row,state = self.map_helper.topological_index_to_occupancy_x_y(int(i))
                self._localization_grid[row,col,0] += counts[x] / (self.question_qty - 0)
                self._localization_grid[row,col,state+1] += counts[x] / (self.question_qty - 0)
                x+=1
                # indices = np.where(current_map_raw[:, 0] == i)
                # values = current_map_raw[indices][:, 1]
                # average = np.average(values)
                # count_ind = np.where(unique_elements == i)
                
                # col,row,state = self.map_helper.topological_index_to_occupancy_x_y(int(i))
                # self._localization_grid[row,col,0] += average * counts[count_ind]
                # self._localization_grid[row,col,state+1] += average * counts[count_ind]

        self._localization_grid = self._localization_grid / self._localization_grid.max()





        # self.vqa_features.data = self.vqa_features.data[0:3] 
        # question_answers_indexes = []
        # question_answers_accs = []
        # self.get_logger().debug(f'must be 3 {len(self.vqa_features.data)}')
        # for i in range(len(self.vqa_features.data)):
            

        #     ind = np.where(self.map_helper.topological_map['q_a'] == self.vqa_features.data[i])
        #     # we keep only coincidences in the current question 
        #     ind = ind[0][np.where(ind[1] == i)]
            
        #     # we keep the topological indexes where there is a coincidence :
        #     current_question_indexes = self.map_helper.topological_map['index'][np.unique(ind)] 
        #     current_question_acc = []
 
        #     # we extract the accuracy for each one of them (acc of question times acc of map)            
        #     for index in np.unique(ind):
        #         acc_ind = np.where(self.map_helper.topological_map['q_a'][index][i] == self.vqa_features.data[i])            
        #         acc = acc_ind[0].size / np.nonzero(self.map_helper.topological_map['q_a'][index][i])[0].size              
        #         current_question_acc.append(acc)

        #     question_answers_indexes.extend(current_question_indexes.tolist())
        #     question_answers_accs.extend(current_question_acc)

                    
        #     current_map_raw = np.transpose(np.array([question_answers_indexes,question_answers_accs]))
            
        #     # there are repeated indexes
        #     unique_elements, counts = np.unique(current_map_raw[:, 0], return_counts=True)

        #  # Iterate over the unique elements
        # x=0
   
        # for i in unique_elements:

        #     col,row,state = self.map_helper.topological_index_to_occupancy_x_y(int(i))
        #     self._localization_grid_variant[row,col,0] += counts[x] / (self.question_qty -2)
        #     self._localization_grid_variant[row,col,state+1] += counts[x] / (self.question_qty -2)
        #     x+=1


        # self._localization_grid_variant = self._localization_grid_variant / self._localization_grid_variant.max()
        # ind = np.unravel_index(np.argmax(self._localization_grid[:,:,0], axis=None), self._localization_grid[:,:,0].shape)
 
        # self._localization_grid[ind[0]-20:ind[0]+20,ind[1]-20:ind[1]+20,0] = 1.0
            





        # question_answers_indexes = []
        # question_answers_accs = []

        # for i in range(len(self.vqa_features.data)):
        #     # 'refrigerator' 
        #     ind = np.where(self.map_helper.topological_map['q_a'] == self.vqa_features.data[i])
        #     ind = ind[0][np.where(ind[1] == i)]
            
        #     # we keep the topo indexes where there is a coincidence :
        #     current_question_indexes = self.map_helper.topological_map['index'][np.unique(ind)] 
        #     current_question_acc = []
 
        #     # we extract the accuracy for each one of them (acc of question times acc of map)            
        #     for index in np.unique(ind):                
        #         current_question_acc.append((self.map_helper.topological_map['acc'][index][i].max()/100.0) * (self.vqa_features.acc[i]/100.0))
        #         self.get_logger().debug(f"current_accs before max {self.map_helper.topological_map['acc'][index][i]}")
        #     self.get_logger().debug(f'current_accs nd idexes sizes : {len(current_question_acc)} / {len(current_question_indexes.tolist())}')
        #     self.get_logger().debug(f'current_accs nd idexes types : {type(current_question_acc)} / {type(current_question_indexes.tolist())}')
        #     self.get_logger().debug(f'current_accs nd idexes content : {current_question_acc} / {current_question_indexes.tolist()}')
        #     question_answers_indexes.extend(current_question_indexes.tolist())
        #     question_answers_accs.extend(current_question_acc)
        # self.get_logger().debug(f'current_accs size {len(current_question_acc)}')


            
        # current_map_raw = np.transpose(np.array([question_answers_indexes,question_answers_accs]))
        
        # # there are repeated indexes
        # unique_elements, counts = np.unique(current_map_raw[:, 0], return_counts=True)




        # # Iterate over the unique elements
        # for i in unique_elements:

        #     indices = np.where(current_map_raw[:, 0] == i)
        #     values = current_map_raw[indices][:, 1]
        #     average = np.average(values)
        #     count_ind = np.where(unique_elements == i)
            
        #     col,row,state = self.map_helper.topological_index_to_occupancy_x_y(int(i))
        #     self._localization_grid[row,col,0] += average * counts[count_ind]
        #     self._localization_grid[row,col,state+1] += average * counts[count_ind]
            

        # self._localization_grid = self._localization_grid / self._localization_grid.max()
        # ind = np.unravel_index(np.argmax(self._localization_grid[:,:,0], axis=None), self._localization_grid[:,:,0].shape)
 
        # self._localization_grid[ind[0]-20:ind[0]+20,ind[1]-20:ind[1]+20,0] = 1.0


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

        # self._localization_grid = np.full(
        #     shape=(self.map_helper.occupancy_map.info.height,
        #            self.map_helper.occupancy_map.info.width,
        #            self.state_qty+1),
        #     fill_value=1/((self.state_qty+1)*self.map_helper.occupancy_map.info.height*
        #                 self.map_helper.occupancy_map.info.width))
        self._localization_grid = np.full(
            shape=(self.map_helper.occupancy_map.info.height,
                   self.map_helper.occupancy_map.info.width,
                   self.state_qty+1),
            fill_value= 1/ (self.map_helper.occupancy_map.info.height *
                   self.map_helper.occupancy_map.info.width *
                   (self.state_qty+1)))
        self._localization_grid_variant = np.full(
            shape=(self.map_helper.occupancy_map.info.height,
                   self.map_helper.occupancy_map.info.width,
                   self.state_qty+1),
            fill_value= 1/ (self.map_helper.occupancy_map.info.height *
                   self.map_helper.occupancy_map.info.width *
                   (self.state_qty+1)))

        # self._localization_grid[116,192,0] = 1.0

        self.get_logger().info("grid initialized")

 
        return True

   
    def _calculate_sigma(self):
        pass
    

    def publish_pose(self):
             
        ind = np.unravel_index(np.argmax(self._localization_grid, axis=None), self._localization_grid.shape)
        x, y = self.map_helper._get_world_x_y(ind[1], ind[0])
        theta = self.map_helper._undiscretize_angle(ind[2])
        
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        q = self.map_helper._quaternion_from_euler(0.0, 0.0, theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        self.pose_publisher.publish(msg)

        # ind = np.unravel_index(np.argmax(self._localization_grid_variant, axis=None), self._localization_grid_variant.shape)
        # x, y = self.map_helper._get_world_x_y(ind[1], ind[0])
        # theta = self.map_helper._undiscretize_angle(ind[2])
        
        # msg = PoseWithCovarianceStamped()
        # msg.header.frame_id = 'map'
        # msg.header.stamp = self.get_clock().now().to_msg()

        # msg.pose.pose.position.x = x
        # msg.pose.pose.position.y = y
        # msg.pose.pose.position.z = 0.0

        # q = self.map_helper._quaternion_from_euler(0.0, 0.0, theta)
        # msg.pose.pose.orientation.x = q[0]
        # msg.pose.pose.orientation.y = q[1]
        # msg.pose.pose.orientation.z = q[2]
        # msg.pose.pose.orientation.w = q[3]

        # self.pose_publisher_variant.publish(msg)
       
        return True
        

 
    def localization_algorithm(self):
        
        t1 = self.get_clock().now()
        self.perception_update()
        t2 = self.get_clock().now()
        self.publish_pose()
        self.get_logger().info(f'time on motion update :{t2-t1}')

   
    # CALLBACKS FUNCTIONS:
 




def main(args=None):
    rclpy.init(args=args)

    topo_node = TopologicalLocalization()

    rclpy.spin(topo_node)


    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

