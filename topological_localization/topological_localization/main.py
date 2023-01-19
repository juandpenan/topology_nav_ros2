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
from scipy import ndimage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
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

        # it is the number the map(gridmap) shape will be divided by
        self.kernel_scale = 8 # self.get_parameter('kernel_scale').get_parameter_value().double_value
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        # m/pix
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.__pkg_folder = str(pathlib.Path(__file__).parent.resolve()).removesuffix("/topological_localization")
        self.map_folder = os.path.join(get_package_share_directory('topological_mapping'),'map3.npy')        
        self.vqa_features = None
        self.image_converter = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_helper = None
        self.odom_pose = None
        self.visualizer = None
        #init prediction variables
        self.previous_x = 0
        self.previous_y = 0
        self.previous_angle = 0
        self.d_increment = 0
        self.angle_increment = 0     
        self.first_run = True
        self.odom_pose = Odometry().pose
        self.odom_pose.pose.position.x = 0.0
        self.odom_pose.pose.position.y = 0.0

        self.timer = self.create_timer(1.0, self.control_cycle)
        # #publishers 
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/markov_pose', 1)
        self.marker_publisher = self.create_publisher(MarkerArray, '/localization_visualizer', 1)
        self.grid_image_publisher = self.create_publisher(Image, '/localization_grid',1)
        #subscribers
        # self.create_subscription(VisualFeatures,"/vqa/features",self.vqa_callback,1)
        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)  
   
        self.tss = ApproximateTimeSynchronizer([Subscriber(self, Odometry, 'odom'),
                                                Subscriber(self, VisualFeatures, 'features')],
                                               10,
                                               6)
        self.tss.registerCallback(self.feature_callback) 
        self.odom_list = []


    def control_cycle(self):

        if len(self.odom_list) < 2 :
            return        
        self.d_increment, self.angle_increment = self._calculate_increment(self.odom_list)
        
        self.get_logger().info(f'angle increment {self.angle_increment} dist increment {self.d_increment}')
        self.localization_algorithm()        

    def map_callback(self,msg):
        self.map_helper = TopologicalMap(msg,self.state_qty,self.question_qty)
        self.get_logger().info("loading topological map .. ")
        self.get_logger().info('map folder path  ' + self.map_folder)
        self.map_helper.load_map(self.map_folder)      
        self.init_localization_grid()
        self.visualizer = Visualizer(self.map_helper)

    def grid_to_msg(self):
        fig = Figure(figsize=(20, 15), dpi=5)
        canvas = FigureCanvas(fig)
        # Do some plotting here
        ax = fig.add_subplot(111)
        ax.pcolor(self._localization_grid[:,:,0]*10000000000.0, cmap=plt.cm.get_cmap("Blues"))
        # Retrieve a view on the renderer buffer
        canvas.draw()
        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        X = np.asarray(buf)       
        return self.image_converter.cv2_to_imgmsg(X)

    def feature_callback(self,odom_msg,vqa_msg):
        if self.map_helper == None:
            return
        
        self.vqa_features = vqa_msg
        self.odom_pose = odom_msg.pose

        self.odom_list.append(self.odom_pose)
        if len(self.odom_list) > 2:
            self.odom_list.pop(0)

   
        # if (self.first_run):
        #     self.previous_x = odom_msg.pose.pose.position.x
        #     self.previous_y = odom_msg.pose.pose.position.y
        #     self.previous_angle = odom_msg.pose.pose.orientation.z
        # x = odom_msg.pose.pose.position.x
        # y = odom_msg.pose.pose.position.y
        # angle = odom_msg.pose.pose.orientation.z
        # self.d_increment = math.sqrt((x - self.previous_x)**2 + (y - self.previous_y)**2)
        # self.angle_increment = math.sqrt((angle - self.previous_angle)**2)
        # self.previous_x = odom_msg.pose.pose.position.x
        # self.previous_angle = odom_msg.pose.pose.orientation.z
        # self.previous_y = odom_msg.pose.pose.position.y
        # self.first_run = False 
        # self.localization_algorithm()
        # self.init_localization_grid()       
        return
    def init_localization_grid(self):

        self._localization_grid = np.full(
            shape=(self.map_helper.topological_map.shape[0],
                   self.map_helper.topological_map.shape[1],
                   self.map_helper.topological_map.shape[2]+1),
            fill_value=1/((self.state_qty+1)*self.map_helper.topological_map.shape[0]*
                        self.map_helper.topological_map.shape[1]))

        self.get_logger().info("grid initialized") 
 
        return True

    def convolve_1d(self,kernel_1d):

        for x in range(self._localization_grid.shape[0]):
            for y in range(self._localization_grid.shape[1]):            
                self._localization_grid[x,y,1:] = ndimage.convolve1d(self._localization_grid[x,y,1:], kernel_1d, mode="constant")       
        return True
   
    def _calculate_1d_kernel_size(self):
        size = int(self.map_helper.topological_map.shape[2]/ self.kernel_scale)
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
        height = int(self.map_helper.topological_map.shape[0] / self.kernel_scale)
        width = int(self.map_helper.topological_map.shape[1] / self.kernel_scale) 
        return (height,width)
    def _calculate_2d_kernel_center(self,odom_pose,kernel_shape,is_centered=False):
        kernel_resolution = ((self.map_helper.topological_map.shape[0] * self.map_resolution / kernel_shape[0]) + \
                            (self.map_helper.topological_map.shape[1] * self.map_resolution / kernel_shape[1]))/2
 

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
        # Define the kernel size
        n = k_size

        # Create an empty kernel
        kernel = np.zeros(n)

        # Calculate the values of the Gaussian distribution at each element of the kernel
        for x in range(n):
            kernel[x] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - center)**2 / (2 * sigma**2)))

        # Normalize the kernel so that the values sum to 1
        kernel = kernel / np.sum(kernel)

        return kernel
        
    def _2d_gaussian_kernel(self,k_size = (10,10) ,sig = [1,1], center = (2,2)):

        # Define the kernel size
        n_h = k_size[0]
        n_w = k_size[1]

        # Define the standard deviation of the Gaussian distribution for each axis
        sigma_h = sig[0]
        sigma_w = sig[1]

        # Define the center of the kernel
        center_h = center[0]
        center_w = center[1]


        # Create an empty kernel
        kernel = np.zeros((n_h, n_w))

        # Calculate the values of the Gaussian distribution at each element of the kernel
        for h in range(n_h):
            for w in range(n_w):
                try:
                    kernel[h, w] = (1 / (np.sqrt(2 * np.pi) * sigma_h * sigma_w)) * np.exp(-(((h - center_h)**2 / (2 * sigma_h**2)) + ((w - center_w)**2 / (2 * sigma_w**2))))
                except ValueError:
                    self.get_logger().error('error with kernel')  

        # Normalize the kernel so that the values sum to 1   
        kernel = kernel / np.sum(kernel)

        return kernel
    def prediction_step(self,delta_distance,delta_theta):
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
        self._localization_grid[:,:,0] = ndimage.convolve(self._localization_grid[:,:,0], gauss_kernel_2d, mode='constant')
        self.convolve_1d(gauss_kernel_1d)
        # ind = np.unravel_index(np.argmax(self._localization_grid[:,:,0], axis=None), self._localization_grid[:,:,0].shape)   
        # ind_1d = np.argmax(self._localization_grid[192,116,1:])        
        self.get_logger().debug(f"center 2d {center_2d}")    
        return
    def normalize_grid(self):
        self._localization_grid = self._localization_grid / np.sum(self._localization_grid)
    def perception_update(self):
        for row in range(self.map_helper.topological_map.shape[0]):
            for colum in range(self.map_helper.topological_map.shape[1]):
                sum = 0
                for state in range(self.map_helper.topological_map.shape[2]):                                   
                    for i in range(len(self.vqa_features.data)):                        
                        if self.map_helper.topological_map[row,colum,state,i][0] != 0:
                            self.get_logger().debug(f'answer: {self.map_helper.topological_map[row,colum,state,i][0]}')
                            for answ in range(self.map_helper.topological_map[row,colum,state,i][0].shape[0]):                                   
                                if self.map_helper.topological_map[row,colum,state,i,0][answ] == self.vqa_features.data[i]:
                                    sum += self.vqa_features.acc[i]        
                                    self._localization_grid[row,colum,0] = sum      
                                    self._localization_grid[row,colum,state+1] = sum                                   
                                else:
                                    continue

        self.normalize_grid()
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
        
        # ind_x_y = np.unravel_index(np.argmax(self._localization_grid, axis=None), self._localization_grid.shape)
        # self.get_logger().info(f'max before{ind_x_y }')

        # self.get_logger().debug(f'sum after{np.sum(self._localization_grid)}')
        t1 = self.get_clock().now()
        self.prediction_step(self.d_increment,self.angle_increment)
        t2 = self.get_clock().now()
        self.get_logger().info(f'time after prediction {(t2-t1)}')        
        # ind_x_y = np.unravel_index(np.argmax(self._localization_grid, axis=None), self._localization_grid.shape)
        # # # self.get_logger().info(f'max after{ind_x_y }')
        # # # # ind_angle =  np.argmax(self._localization_grid[ind_x_y,1:], axis=None)
        # angle = np.pi*2/8 * ind_x_y[2]
        # # # self.map_helper.occupancy_x_y_to_topological_index(ind_x_y[0],ind_x_y[1])}
       
        # x_world,y_world = self.map_helper._get_world_x_y(ind_x_y[1],ind_x_y[0])
  
        # # self.publish_pose(x_world,y_world,angle)
        # # self.publish_pose(x_world,y_world,angle)

       

        # # x,y = self.map_helper._get_occupancy_x_y(self.odom_pose.pose.position.x,self.odom_pose.pose.position.y)
        # # self.get_logger().info("Current costmap pose x:%f y::%f " %(x,y))  
        # # self.get_logger().info("Actual costmap location x:%f y:%f " %(x,y))
        # self.get_logger().debug("Actual pose location x:%f y:%f " %(self.odom_pose.pose.position.x,self.odom_pose.pose.position.y))
        # self.get_logger().debug("Algorithm prediction pose location x:%f y:%f " %(x_world,y_world))
        # msg = self.grid_to_msg()
        # self.grid_image_publisher.publish(msg)
      
        # # msg = self.visualizer.get_msg_from_grid(self._localization_grid,self)
        
        # self.get_logger().debug("published")
   
    # CALLBACKS FUNCTIONS:
 




def main(args=None):
    rclpy.init(args=args)

    topo_node = TopologicalLocalization()

    rclpy.spin(topo_node)


    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

