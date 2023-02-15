import os
import pathlib

import cv2

from nav_msgs.msg import OccupancyGrid

import numpy as np

import rclpy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from topological_mapping.topological_mapping.topological_map import TopologicalMap

from vqa_ros.vqa_ros import VQAModel

class SaveMapToDisc(VQAModel):
    def __init__(self):
        super().__init__()
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
        self.question_depth = self.get_parameter(
            'max_images_per_state').get_parameter_value().integer_value

        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)

    def map_callback(self, map_msg):
        self.map_msg = map_msg
        map = TopologicalMap(map_msg, self.state_qty, self.question_qty)

        rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix(
            '/topological_mapping/topological_mapping') + '/data/'

        rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix(
            '/topological_mapping/topological_mapping') + '/real/data/'

        state = 0

        for subdir, dirs, files in os.walk(rootdir):
            try:
                index = int(subdir.removeprefix(rootdir))
            except Exception:
                self.get_logger().error('Fail to convert '
                                        + subdir.removeprefix('/home/juan/Workspaces/phd_ws/src/topological_local/topological_mapping/data/') + 'to int')
                continue

            dt = np.dtype({'names': ['index', 'q_a', 'acc'],
                           'formats': ['i8',
                                       '('+str(self.question_qty)+','+str(self.question_depth)+')'
                                       + 'U16',
                                       '('+str(self.question_qty)+','+str(self.question_depth)+')'
                                       + 'f']})

            glob = np.zeros(shape=1, dtype=dt)

            # for each state folder
            for subdir, dirs, files in os.walk(subdir):
                self.get_logger().info('saving in index :'+str(index))
                global_conf = np.zeros(shape=(self.question_depth,), dtype='f')
                global_answers = np.zeros(shape=(self.question_depth,), dtype='U16')
                qdpth = 0
                # for each image in each state folder                 
                for file in files:
                    image = cv2.imread(subdir+'/'+file)                                                                    
                    answers = []
                    conf = []
                    q_qty = 0
                    for question in self.questions:                          
                        answer, confi = self._plot_inference_qa(image, question)
                        self.get_logger().info(f'appending in array {q_qty} {qdpth}')
                        glob['index'][0] = index
                        glob['q_a'][0][q_qty][qdpth] = answer
                        glob['acc'][0][q_qty][qdpth] = confi
                        q_qty += 1
                        # answers.append(answer)
                        # conf.append(confi)
                #     global_conf[i] = conf
                #     global_answers.append[i] = answers
                    qdpth += 1
                # global_conf = np.array(global_conf).T
                # global_answers = np.array(global_answers).T

                if map.is_map_empty(index=index):
                    map.update_map(glob_features=glob, index=index)
                    self.get_logger().info('map updated cuz it was empty')
            self.get_logger().info('savig map')
            map.save_map(path=str(pathlib.Path(__file__).parent.resolve()).removesuffix(
                '/topological_mapping/topological_mapping')+'/map/map6',
                         questions=self.questions, name='map6')
            self.get_logger().info('map saved')

            state += 1

        self.get_logger().info('Finished')
        self.destroy_node()
        rclpy.shutdown()
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SaveMapToDisc()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
