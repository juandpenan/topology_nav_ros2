from vqa_ros.vqa_ros import VQAModel
import pathlib
import os
import rclpy
from rclpy.node import Node
import cv2
from statistics import mean
import yaml
from ament_index_python.packages import get_package_share_directory
from topological_mapping.topological_mapping.topological_map import TopologicalMap
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy


class SaveMapToDisc(VQAModel):
    def __init__(self):
        super().__init__()
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability= QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        #paraameters
        self.declare_parameter('question_qty', 10)  
        self.declare_parameter('state_qty', 8)
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty  = self.get_parameter('question_qty').get_parameter_value().integer_value 

        self.create_subscription(OccupancyGrid,"/map",self.map_callback,qos_profile=map_qos_profile)

    def map_callback(self,map_msg):
        
        map = TopologicalMap(map_msg,self.state_qty,self.question_qty)
        
        rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix("/topological_mapping/topological_mapping") + "/data/"
        for subdir, dirs, files in os.walk(rootdir):
            try:        
                index = int(subdir.removeprefix(rootdir))
            except:
                    self.get_logger().error("Fail to convert "+subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/")+" to int")
                    continue
                    

            for subdir, dirs, files in os.walk(subdir):
                print("saving in index :"+str(index))                    
                for file in files:
                    
                    image = cv2.imread(subdir+"/"+file)                      
                    answers = []
                    conf = []                    
                    for question in self.questions:                          
                        answer,confi = self._plot_inference_qa(image,question)                           
                        answers.append(answer)
                        conf.append(confi)
                    
                    if map.is_map_empty(index=index):                            
                        map.update_map(features=answers,acc=conf,index=index)
                        self.get_logger().info("map updated cuz it was empty")
                    else:
                        a = map.consult_map(index=index)                 
                        if mean(conf) > a[:,:,1].mean():
                                                            
                            map.update_map(features=answers,acc=conf,index=index)
                            self.get_logger().info("map updated cuz mean higer")
                            
        self.get_logger().info("savig map")
        map.save_map(questions = self.questions)
        self.get_logger().info("map saved")                

        return True








def main(args=None):
    rclpy.init(args=args)
    node = SaveMapToDisc()

    rclpy.spin(node)


    node.destroy_node()
    rclpy.shutdown()

    # ros2_serial_example_directory = os.path.join(
    # get_package_share_directory('vqa_ros'))

    # param_config = os.path.join(ros2_serial_example_directory,
    #                             'params.yaml')

    # with open(param_config, 'r') as f:
    #     params = yaml.safe_load(f)['vqa_node']['ros__parameters']

    # question_list = params["questions"]
    # vqa = VQAModel()
    # vqa.create_subscription(OccupancyGrid,"/map",map_cb,1)
    
    # map = TopologicalMap(msg,8,10)

    # rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix("/topological_mapping/topological_mapping") + "/data/"
    # # /home/juan/Workspaces/phd_ws/src/topological_localization/topological_mapping/topological_mapping/data/

    # for subdir, dirs, files in os.walk(rootdir):
    #     try:        
    #         index = int(subdir.removeprefix(rootdir))
    #     except:
    #          print("Fail to convert "+subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/")+" to int")
    #          continue
                

    #     for subdir, dirs, files in os.walk(subdir):
    #         print("saving in index :"+str(index))                    
    #         for file in files:
                
    #             image = cv2.imread(subdir+"/"+file)                      
    #             answers = []
    #             conf = []                    
    #             for question in question_list:
    #                 print(question)
    #                 answer,confi = vqa._plot_inference_qa(image,question)
    #                 print(answer)
    #                 answers.append(answer)
    #                 conf.append(confi)
                
    #             if map.is_map_empty(index=index):
    #                 print(answers)
    #                 map.update_map(features=answers,acc=conf,index=index)
    #                 print("map updated cuz it was empty")
    #             else:
    #                 a = map.consult_map(index=index)                 
    #                 if mean(conf) > a[:,:,1].mean():
    #                     print(answers)                                
    #                     map.update_map(features=answers,acc=conf,index=index)
    #                     print("map updated cuz mean higer")
                        
    #     print("savig map")
    #     map.save_map()
    #     print("map saved")                
        # except:
        #     print("Fail to convert "+subdir.removeprefix("/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/")+" to int")




if __name__ == '__main__':
    main()



