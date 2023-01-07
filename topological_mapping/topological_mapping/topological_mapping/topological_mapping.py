import rclpy
import numpy as np
import cv2
import os
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
from statistics import mean

import pathlib

# from vqa_ros.vqa_ros import VQAModel

class TopologicalMapping(Node):
    def __init__(self):
        super().__init__('topological_mapping')        
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability= QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.declare_parameter('question_qty', 10)  
        self.declare_parameter('state_qty', 8)  
        self.declare_parameter('max_images_per_state', 10)  
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty  = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.max_images_per_state  = self.get_parameter('max_images_per_state').get_parameter_value().integer_value
        self.map = None
        self.previous_state =None
        self.state_counter = 0
        self.features = VisualFeatures()
        self.pose = PoseWithCovarianceStamped() 
        self._update_counter = 0
        self.image_converter = CvBridge() 
        self.first_run = True       
        self._map_folder = str(pathlib.Path(__file__).parent.resolve()).removesuffix("/topological_mapping") + "/testing/"
        #publishers
        self.marker_publisher = self.create_publisher(MarkerArray,"/localization_visualizer",map_qos_profile)
        #subscribers
        self.create_subscription(OccupancyGrid,"/map",self.map_callback,qos_profile=map_qos_profile)
       
      
        self.tss = ApproximateTimeSynchronizer([Subscriber(self,Image,'image'),
                                                Subscriber(self,Odometry,'odom')],
                                                10,
                                                0.25)
        self.tss.registerCallback(self.store_data_cb)
        self.get_logger().info("params: %i ,%i"%(self.state_qty,self.question_qty))
        self.get_logger().info("topological_mapping initialized")


    def store_data_cb(self,image_msg,odom_msg):
        try:
            self.map.state_qty
        except:
            self.get_logger().warning("map has not been initialized yet")
            return
        else:
            if self.first_run:
                self.get_logger().info("first run") 
                self.previous_state = str(self.map.pose_to_index(pose=odom_msg))

            state = str(self.map.pose_to_index(pose=odom_msg))
            self.get_logger().info("previous and current: %s / %s"%(self.previous_state,state)) 
            if state == self.previous_state:                
                if self.state_counter < (self.max_images_per_state):
                    self.state_counter+=1
                    self.__save_image_to_disc(image_msg,odom_msg)                 
                else:
                    self.get_logger().info("this state already reached limit images")
            else:                
                self.state_counter = 0 
                self.__save_image_to_disc(image_msg,odom_msg)
        
            self.previous_state = str(self.map.pose_to_index(pose=odom_msg))            
            self.first_run = False      
        return True
    def __save_image_to_disc(self,img,pose):
        folder_name = str(self.map.pose_to_index(pose=pose))
        _path = os.path.join(self._map_folder +folder_name)
        os.makedirs(_path, exist_ok=True)
        image =  self.image_converter.imgmsg_to_cv2(img)      	
        cv2.imwrite(_path+"/"+str(self.get_clock().now().nanoseconds)+".jpg", image)
        
    def save_map_cb(self,msg):
        if self.map is None or self.pose is None:
            return
        elif msg.data == 1:
            self.map.save_map()
            print("map saved")
        else:
            return

    def map_callback(self,msg):
        self.map =TopologicalMap(msg,self.state_qty,self.question_qty)

  
            
    def __generate_map_from_images(self):

        rootdir = "/home/juan/Workspaces/phd_ws/src/topological_localization/topological_mapping/data/"
            

        for subdir, dirs, files in os.walk(rootdir):
            try:
                
                index = int(subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_localization/topological_mapping/data/"))
                
                for subdir, dirs, files in os.walk(subdir):
                    print("saving in index :"+str(index))                    
                    for file in files:
                        image = cv2.imread(subdir+"/"+file)                      
                        answers = []
                        conf = []
                        for question in question_list:
                            answer,confi = vqa_model._plot_inference_qa(image,question)
                            answers.append(answer)
                            conf.append(confi)
                        
                        if self.map.is_map_empty(index=index):
                            print(answers)
                            self.map.update_map(features=answers,acc=conf,index=index)
                            print("map updated cuz it was empty")
                        else:
                            a = self.map.consult_map(index=index)                 
                            if mean(conf) > a[:,:,1].mean():
                                print(answers)                                
                                self.map.update_map(features=answers,acc=conf,index=index)
                                print("map updated cuz mean higer")
                                
                print("savig map")
                self.map.save_map()
                print("map saved")                
            except:
                print("Fail to convert "+subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/")+" to int")
            
     
     
    def pose_callback(self,msg):
        self.pose = msg
       
    
    def image_callback(self,msg):
        if self.map is None or self.pose is None:
            print("returning from image callback")
            return
        _path = "/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/"        
        folder_name = str(self.map.pose_to_index(pose=self.pose))
        _path = os.path.join(_path+folder_name)
        os.makedirs(_path, exist_ok=True)

        image =  self.image_converter.imgmsg_to_cv2(msg)
        print("writing image")       	
        cv2.imwrite(_path+"/"+str(self.get_clock().now().nanoseconds)+".jpg", image)
        return

     
    def features_callback(self,msg):
         if self.map is None or self.pose is None:
            print("returning")
            return
         elif self.map.is_map_empty(pose=self.pose):
            self.get_logger().info("Updating map")            
            self.map.update_map(msg.data,msg.acc,pose=self.pose)
            index = self.map.pose_to_index(pose=self.pose)
            pos = self.map.index_to_pose(index=index)

            self.visualizer.publish_rviz(pose = pos)
         else:
           self.get_logger().info("Map is already updated")





        

       

        
 


     




def main(args=None):
    rclpy.init(args=args)

    topo_node = TopologicalMapping()

    rclpy.spin(topo_node)


    topo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

