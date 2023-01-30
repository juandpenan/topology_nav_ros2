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
        self.kernel_scale = 8 # self.get_parameter('kernel_scale').get_parameter_value().double_value
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.question_depth = self.get_parameter('max_images_per_state').get_parameter_value().integer_value
        # m/pix
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value

        self.__pkg_folder = str(pathlib.Path(__file__).parent.resolve()).removesuffix('/topological_localization')
        self.map_folder = os.path.join(get_package_share_directory('topological_mapping'), 'map2.npy')
        self.image_map_folder = os.path.join(get_package_share_directory('topological_mapping'), 'map3.jpg')

        self.image_converter = CvBridge()        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.map_helper = None
        #init prediction variables
        self.odom_pose = Odometry().pose
        self.odom_pose.pose.position.x = 0.0
        self.odom_pose.pose.position.y = 0.0
        self.d_increment = self.angle_increment = 0.0
        self.timer = self.create_timer(3.0, self.control_cycle)
        self.odom_list = []
        # #publishers 
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/markov_pose', 1)
        self.grid_publisher = self.create_publisher(OccupancyGrid, '/motion_update/localization_grid', 1)
        #subscribers

        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)
        # remapped from your odom topic(ground_truth_odom) to /odom
        self.create_subscription(Odometry,
                                 '/odom',
                                 self.odom_callback,
                                 1)
        self.broadcast_map()

    def control_cycle(self):
        
        self.odom_list.append(self.odom_pose)
        if len(self.odom_list) > 2:
            self.odom_list.pop(0)

        if len(self.odom_list) < 2:
            return

        self.get_logger().info('executing algorithm')
        self.d_increment, self.angle_increment = self._calculate_increment(self.odom_list)

        ind = np.unravel_index(np.argmax(self._localization_grid, axis=None), self._localization_grid.shape)
        self._localization_grid[ind] = 1.0
        self.localization_algorithm()

        image = self.grid_to_img()
        map = self.img_to_occupancy(image)

        self.grid_publisher.publish(map)



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

    def img_to_occupancy(self,image):

        if self.map_helper == None:
            return

        image = image * 100
        map = self.map_helper.occupancy_map
        data = np.array(image.flatten(), dtype=np.int8)
        map.data = data.tolist()

        return map


    def odom_callback(self,odom_msg):

        if self.map_helper == None:
            return        
 
        self.odom_pose = odom_msg.pose



        return True

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
            fill_value= 0.2)

        self._localization_grid[116,192,0] = 1.0

        self.get_logger().info("grid initialized")

 
        return True

    def convolve_1d(self,kernel_1d):

        _ = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="same"), axis=-1, arr=self._localization_grid[:,:,1:])
        self._localization_grid[:,:,1:] = _

        return True      

   
    def _calculate_1d_kernel_size(self): 

        size = int(self.state_qty / self.kernel_scale)
        return size
    def _calculate_1d_kernel_center(self,odom_pose,kernel_shape,is_centered=False):

        kernel_resolution = 2*np.pi / kernel_shape  #rad/div
        angle = self.map_helper._quaternion_to_euler(odom_pose.pose.orientation)[-1]
        if angle < 0:
            angle = angle + 2*np.pi
        elif angle > 2 * np.pi:
            angle =  2 * np.pi
        if is_centered:
            center = int(round(kernel_shape/2))
        else:    
            center = int(round(angle/kernel_resolution)) 
        return center
    
    def _calculate_2d_kernel_size(self):

        height = int(self.map_helper.occupancy_map.info.height / self.kernel_scale)
        width = int(self.map_helper.occupancy_map.info.width / self.kernel_scale) 
        return (height,width)

    def _calculate_2d_kernel_center(self,odom_pose,kernel_shape,is_centered=False):

        kernel_resolution = ((self.map_helper.occupancy_map.info.height * self.map_resolution / kernel_shape[0]) + \
                            (self.map_helper.occupancy_map.info.width * self.map_resolution / kernel_shape[1]))/2
 

        if is_centered:           
            h = int(kernel_shape[0]/2)
            w = int(kernel_shape[1]/2)
        else:            
            h = int(round((odom_pose.pose.position.y - self.map_helper.occupancy_map.info.origin.position.y ) / kernel_resolution))
            w = int(round((odom_pose.pose.position.x - self.map_helper.occupancy_map.info.origin.position.x ) / kernel_resolution))
            
        return (h,w)

    def _calculate_sigma(self):
        return
    def _1d_gaussian_kernel(self,k_size = 5,sigma = 1.0,center = 2):

        x = np.arange(k_size)
        
        kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - center) ** 2 / (2 * sigma ** 2)))
       
        kernel = kernel / np.sum(kernel)     

        return kernel
        
    def _2d_gaussian_kernel(self,k_size = (10,10) ,sig = [1,1], center = (2,2)):
        # Define the kernel size
        n_h, n_w = k_size

        # Define the standard deviation of the Gaussian distribution for each axis
        sigma_h, sigma_w = sig

        # Define the center of the kernel
        center_h, center_w = center

        # Create 2D coordinate arrays for the kernel using np.mgrid
        y, x = np.mgrid[:n_h, :n_w]

        # Calculate the values of the Gaussian distribution at each element of the kernel
        kernel = (1 / (np.sqrt(2 * np.pi) * sigma_h * sigma_w)) * np.exp(-(((x - center_w) ** 2 / (2 * sigma_w ** 2)) + ((y - center_h) ** 2 / (2 * sigma_h ** 2))))

        # Normalize the kernel so that the values sum to 1
        kernel = kernel / np.sum(kernel)

        return kernel

    def motion_update(self,delta_distance,delta_theta):

        self.get_logger().debug(f'delta distance{delta_distance}')         
        if delta_distance <= 0.1:
            centered_2d = True
        else:
            centered_2d = False
        if delta_theta <= 0.0872665:
            centered_1d = True
        else:
            centered_1d = False
        kernel_shape_2d =  self._calculate_2d_kernel_size()
        kernel_shape_1d = self._calculate_1d_kernel_size()

        center_2d = self._calculate_2d_kernel_center(self.odom_pose,kernel_shape_2d,is_centered=centered_2d)    
        center_1d = self._calculate_1d_kernel_center(self.odom_pose,kernel_shape_1d,is_centered=centered_1d)
         
        gauss_kernel_2d =  self._2d_gaussian_kernel(kernel_shape_2d,center=center_2d)
        gauss_kernel_1d = self._1d_gaussian_kernel(kernel_shape_1d,center=center_1d)

        self._localization_grid[:,:,0] = signal.fftconvolve(self._localization_grid[:,:,0], gauss_kernel_2d, mode='same')
        self.convolve_1d(gauss_kernel_1d)       
        self.get_logger().debug(f"center 2d {center_2d}")

        return True
 

    def publish_pose(self,x,y,theta,frame="map",covariance=1.0):
        msg = PoseWithCovarianceStamped()
        t = TransformStamped()
        msg.pose.covariance[0] = covariance
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        q = self.map_helper._quaternion_from_euler(0,0,theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        self.pose_publisher.publish(msg)
        try:
            robot2odom = self.tf_buffer.lookup_transform(
                        "base_footprint",
                        "odom",
                        rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info("got into exception")
            self.get_logger().info(
                f'Could not transform base_footprint to odom: {ex}')
            return
        self.get_logger().info("transform ok")
        map2robot = TransformStamped()
        map2robot.transform.translation.x = x
        map2robot.transform.translation.y = y
        map2robot.transform.translation.z = 0.0
        map2robot.transform.rotation.x = q[0]
        map2robot.transform.rotation.y = q[1]
        map2robot.transform.rotation.z = q[2]
        map2robot.transform.rotation.w = q[3]
        # euler_robot2odom = tf_transformations.euler_from_quaternion([robot2odom.transform.rotation.x,
        #                                                             robot2odom.transform.rotation.y,
        #                                                             robot2odom.transform.rotation.z,
        #                                                             robot2odom.transform.rotation.w])
        map2robot_transform = tf2_kdl.transform_to_kdl(map2robot)
        # PyKDL.Frame(PyKDL.Rotation.RPY(0,0,theta),
        #                             PyKDL.Vector(x,y,0.0))
        robot2odom_transform = tf2_kdl.transform_to_kdl(robot2odom)
        # PyKDL.Frame(PyKDL.Rotation.RPY(euler_robot2odom[0],euler_robot2odom[1],euler_robot2odom[2]),
        #                             PyKDL.Vector(robot2odom.transform.translation[0],
        #                                          robot2odom.transform.translation[1],
        #                                          robot2odom.transform.translation[2]))
        #
        odom2map = np.dot(map2robot_transform,robot2odom_transform)
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = odom2map[0, 3]
        t.transform.translation.y = odom2map[1, 3]
        t.transform.translation.z = odom2map[2, 3]
        t.transform.rotation.w = np.sqrt(1 + odom2map[0, 0] + odom2map[1, 1] + odom2map[2, 2]) / 2
        t.transform.rotation.x = (odom2map[2, 1] - odom2map[1, 2]) / (4 * t.transform.rotation.w)
        t.transform.rotation.y = (odom2map[0, 2] - odom2map[2, 0]) / (4 * t.transform.rotation.w)
        t.transform.rotation.z = (odom2map[1, 0] - odom2map[0, 1]) / (4 * t.transform.rotation.w)
        self.tf_broadcaster.sendTransform(t)
        return

    def _calculate_increment(self,msg_buffer):

        intial_distance = math.sqrt(msg_buffer[0].pose.position.x ** 2 + 
                                    msg_buffer[0].pose.position.y ** 2 +
                                    msg_buffer[0].pose.position.z ** 2)

        current_distance = math.sqrt(msg_buffer[1].pose.position.x ** 2 + 
                                     msg_buffer[1].pose.position.y ** 2 +
                                     msg_buffer[1].pose.position.z ** 2)

        initial_angle = self.map_helper._quaternion_to_euler(msg_buffer[0].pose.orientation)[-1]
        current_angle = self.map_helper._quaternion_to_euler(msg_buffer[1].pose.orientation)[-1]
        
        distance_increment = abs(current_distance - intial_distance)
        angle_increment = abs(current_angle - initial_angle)

        return distance_increment, angle_increment

    def localization_algorithm(self):
        
        t1 = self.get_clock().now()
        self.motion_update(self.d_increment,self.angle_increment)
        t2 = self.get_clock().now()
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

