import rclpy
import numpy as np
import cv2
import copy
from topological_mapping.topological_mapping.topological_map import TopologicalMap
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from math import pi
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from builtin_interfaces.msg import Duration
from vqa_msgs.msg import VisualFeatures
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from std_msgs.msg import Int16
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
from topological_mapping.topological_mapping.topological_map import TopologicalMap


class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')
        self.get_logger().info("map_visualizer initialized")

        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability= QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        #publishers
        self.marker_publisher = self.create_publisher(MarkerArray,"/localization_visualizer",map_qos_profile)
        self.counter = 0
        self.map_helper= None
        self.amsg = MarkerArray()
        self.visualize_map()
        self.odom_pose = PoseWithCovarianceStamped()
        #subscribers
        self.create_subscription(PoseWithCovarianceStamped,"/markov_pose",self.markov_callback,1)
        self.create_subscription(PoseWithCovarianceStamped,"/map",self.map_callback,1)


    def map_callback(self,msg):
        self.map_helper = TopologicalMap(msg,8,10)
        #TODO CHANGE TO MORE GENERAL PATH
        self.map_helper.load_map("/home/juan/Workspaces/phd_ws/src/topological_localization/topological_mapping/map/map.npy")

    def markov_callback(self,msg):
        self.get_logger().info("Publishing poses!!")
        self.publish_rviz(msg.pose.pose,color=[0.0,1.0,0.0])
        # self.publish_rviz(self.odom_pose.pose,color=[0.0,1.0,0.0])

    def odom_callback(self,msg):
        self.odom_pose= msg.pose
    def sync_callback(self,mrkv_msg,odom_msg):
        
        self.publish_rviz(mrkv_msg.pose,color=[0.0,0.0,1.0])
        self.publish_rviz(odom_msg.pose,color=[0.0,1.0,0.0])


        return
    def pose_visualizer(self,msg):

        index =  self.map_helper.pose_to_index(msg)
        pose = self.map_helper.index_to_pose(index=index)
        self.publish_rviz(pose)




        return
    def __count_indexes(self):
        map_indexes =[]
        for row in range(self.map_helper.topological_map.shape[0]):
            for colum in range(self.map_helper.topological_map.shape[1]):
                for state in range(self.map_helper.topological_map.shape[2]):
                    for question in range(self.map_helper.question_qty):
                        if self.map_helper.topological_map[row,colum,state,question][0] != 0:
                            map_indexes.append([row,colum,state])

                           
        return map_indexes
    def visualize_map(self):
        self.__load_map()
        map_indexes = self.__count_indexes()
        msg = Marker()
        msg.header.frame_id = "map"
        msg.color.a = 1.0
        msg.scale.x = 0.1
        msg.scale.y = 0.01
        msg.scale.z = 0.01 
        msg.type = 0  
        msg.pose.position.z = 0.0
        
        for i in range(len(map_indexes)):   
            self.counter += i      
            msg.id = self.counter  
            msg.color.r = self.set_color(map_indexes[i])            
            x,y = self.map_helper._get_world_x_y(map_indexes[i][1],map_indexes[i][0])
            angle = map_indexes[i][2]*np.pi/8
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.orientation.z = self.map_helper._quaternion_from_euler(0,0,angle)[2]
            self.amsg.markers.append(copy.deepcopy(msg))
        self.marker_publisher.publish(self.amsg)
        self.get_logger().info("published")
        



        return True

    def publish_rviz(self,pose,color=[0.0,1.0,0.0],alpha=1.0):  
        self.counter +=1       
        # pose = pose.pose.pose      
        msg = Marker()
        msg.color.r = color[0]
        msg.color.g = color[1]
        msg.color.b = color[2]
        msg.color.a = alpha
        msg.header.frame_id = "map"
        msg.scale.x = 0.15
        msg.scale.y = 0.05
        msg.scale.z = 0.05
        
        msg.id = self.counter
        msg.type = 0
        msg.pose = pose

        self.amsg.markers.append(copy.deepcopy(msg))

        self.marker_publisher.publish(self.amsg)
    def set_color(self,index):

        x = (np.mean(self.map_helper.topological_map[index[0],index[1],index[2],:,1]))/100.0
        return ((x-0.7)/(1.0-0.7))





def main(args=None):
    rclpy.init(args=args)

    visualizaer_node = MapVisualizer()

    rclpy.spin(visualizaer_node)


    visualizaer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

