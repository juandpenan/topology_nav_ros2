import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import copy
import pickle
from builtin_interfaces.msg import Duration
import multiprocessing as mp
from itertools import repeat

# def process_data(data):
#   result = []
#   for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#       for k in range(data.shape[2]):
#         if k == 0:
#           continue
#         else:
#           result.append(data[i, j, k] * 2)
#   return result

# data = np.random.random((10, 10, 10))

# with mp.Pool(4) as p:
#   result = p.map(process_data, [data[i::4] for i in range(4)])


class Visualizer():

    def __init__(self, map_helper):
        duration = Duration()
        duration.sec = 1
        self.marker_array_msg = MarkerArray()
        self.map_helper = map_helper
        self.marker_msg = Marker()
        self.marker_msg.header.frame_id = 'map'
        self.marker_msg.color.r = 1.0
        self.marker_msg.scale.x = 0.02
        self.marker_msg.scale.y = 0.002
        self.marker_msg.scale.z = 0.002
        self.marker_msg.type = 0
        self.marker_msg.pose.position.z = 0.0
        # self.marker_msg.lifetime = copy.deepcopy(duration)
        self.id = 0

    def _get_msg_from_grid(self, grid, node):
        non_zero_indices = np.transpose(np.nonzero(grid[:, :, 1:]))
        c = 0
        # for row, col, state in non_zero_indices:

         

        #     # if c >= 9:

        #     #     node.get_logger().info('break')
        #     #     break
        #     self.marker_msg.id = self.id
        #     self.marker_msg.color.a = 1.0
        #     self.id += 1
        #     x, y = self.map_helper._get_world_x_y(col, row)
        #     node.get_logger().info(str(state))
        #     angle = self.map_helper._undiscretize_angle(state)
        #     self.marker_msg.pose.position.x = x
        #     self.marker_msg.pose.position.y = y

        #     q = self.map_helper._quaternion_from_euler(0, 0, angle)
            
        #     self.marker_msg.pose.orientation.x = q[0]
        #     self.marker_msg.pose.orientation.y = q[1]
        #     self.marker_msg.pose.orientation.z = q[2]
        #     self.marker_msg.pose.orientation.w = q[3]

        #     self.marker_array_msg.markers.append(copy.deepcopy(self.marker_msg))
        with open('/home/juan/Workspaces/phd_ws/src/topological_localization/topological_localization/resource/msg.pkl', 'rb') as inp:
            msg = pickle.load(inp)

        node.marker_publisher.publish(msg)
            # c += 1

        
        # for row in range(grid.shape[0]):
        #     for colum in range(grid.shape[1]):
        #         for state in range(grid.shape[2]):
        #             if state == 0:
        #                 continue
        #             else:
        #                 node.get_logger().info('working on')
        #                 self.marker_msg.id = self.id
        #                 self.marker_msg.color.a = 1.0
        #                 self.id += 1
        #                 x, y = self.map_helper._get_world_x_y(colum, row)
        #                 angle = (state-1) * np.pi/self.map_helper.topological_map.shape[2]
        #                 self.marker_msg.pose.position.x = x
        #                 self.marker_msg.pose.position.y = y
        #                 self.marker_msg.pose.orientation.z = self.map_helper._quaternion_from_euler(0, 0, angle)[2]

        #                 self.marker_array_msg.markers.append(copy.deepcopy(self.marker_msg))

        #                 node.marker_publisher.publish(self.marker_array_msg)

        return self.marker_array_msg

    def get_msg_from_grid(self, grid, node):
        # answers,confidence = zip(*p.starmap(utils._plot_inference_qa, zip(repeat(image),self.questions))) 

        with mp.Pool(4) as p:
            
            _ = zip(*p.starmap(self._get_msg_from_grid, zip( [grid[i::4] for i in range(4)],repeat(node))))

    def get_color_from_prob(self, prob):
        color = ColorRGBA()
        color.a = 1.0

        color.r = float(1-prob)
        color.g = float(prob)

        return color
    
    # def get_init_grid(self,grid):


