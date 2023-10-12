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
        self.declare_parameter('resume_map_path', '')

        self.resume_map_path = self.get_parameter('resume_map_path').get_parameter_value().string_value
        self.state_qty = self.get_parameter('state_qty').get_parameter_value().integer_value
        self.question_qty = self.get_parameter('question_qty').get_parameter_value().integer_value
        self.question_depth = self.get_parameter(
            'max_images_per_state').get_parameter_value().integer_value

        self.timer.destroy()
        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.map_callback,
                                 qos_profile=map_qos_profile)

    def _save_to_disc(self, map, rootdir):

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

                for q_qty, question in enumerate(self.questions):

                    answers = []
                    confs = []

                    for qdpth, file in enumerate(files):

                        if question.find('*') != -1:
                            question = question.replace('*', glob['q_a'][0][q_qty-1][0])

                        image = cv2.imread(subdir+'/'+file)
                        answer, confi = self._plot_inference_qa(image, question)

                        if not answers and not confs:
                            answers.append(answer)
                            confs.append(confi)

                        else:

                            if answer in answers:
                                confs[answers.index(answer)] = (confs[answers.index(answer)]
                                                                + confi) / 2.0
                            else:
                                answers.append(answer)
                                confs.append(confi)

                    glob['index'][0] = index
                    glob['q_a'][0][q_qty][0: len(answers)] = answers
                    glob['acc'][0][q_qty][0: len(answers)] = confs

                if map.is_map_empty(index=index):
                    map.update_map(glob_features=glob, index=index)
                    self.get_logger().info('map updated cuz it was empty')
            self.get_logger().info('savig map')
            map.save_map(path=str(pathlib.Path(__file__).parent.resolve()).removesuffix(
                '/topological_mapping/topological_mapping')+'/map/map13',
                         questions=self.questions, name='map13')
            self.get_logger().info('map saved')

        return

    def map_callback(self, map_msg):
        self.map_msg = map_msg
        map = TopologicalMap(map_msg, self.state_qty, self.question_qty)

        rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix(
            '/topological_mapping/topological_mapping') + '/data/' + '/experiments/'

        self._save_to_disc(map, rootdir)

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
