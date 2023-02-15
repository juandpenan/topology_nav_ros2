import numpy as np
import cv2
import os
import yaml
import pathlib
from math import pi, atan2, copysign, asin, cos, sin
from geometry_msgs.msg import Pose

    
class TopologicalMap():
    _instance = None

    def __new__(cls, *args):
        if not cls._instance:
            cls._instance = super(TopologicalMap, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, occupancy_map, state_qty, question_qty, question_depth=10):

        self.__SEMANTIC_RESULTS_QTY = 2
        self.__pkg_folder = str(pathlib.Path(__file__).parent.resolve()).removesuffix(
            'topological_mapping/topological_mapping')
  
        self.state_qty = state_qty
        self.question_qty = question_qty
        self.occupancy_map = occupancy_map
    
        try:
            self.occupancy_matrix = np.array(self.occupancy_map.data, dtype=np.float32).reshape(
                self.occupancy_map.info.height, self.occupancy_map.info.width)
            self.occupancy_image = self._occupancy_map_to_image()
        except ValueError:
            print('Error loading the costmap image and occupancy matrix')
        
        # Height (y / rows):
        # Width (x / columns)
        # (y,x,#states per grid,#questions,answer string,prob)
        # self.topological_map = np.zeros((self.occupancy_map.info.height,
        #                                  self.occupancy_map.info.width,
        #                                  self.state_qty, self.question_qty,
        #                                  self.__SEMANTIC_RESULTS_QTY), dtype=object)

        dt = np.dtype({'names': ['index', 'q_a', 'acc'],
                       'formats': ['i8',
                                   '('+str(question_qty)+','+str(question_depth)+')'+'U16',
                                   '('+str(question_qty)+','+str(question_depth)+')'+'f']})

        self.topological_map = np.array([], dtype=dt)

    def __del__(self):
        print('Topological map destroyed')

    def topological_index_to_occupancy_x_y(self, int_index):
        indexes = np.unravel_index(int_index,
                                   shape=(self.occupancy_map.info.height,
                                          self.occupancy_map.info.width,
                                          self.state_qty))
        return indexes[1], indexes[0], indexes[2]

    def occupancy_x_y_to_topological_index(self, occupancy_x, occupancy_y, angle):

        dis_angle = self._discretize_angle(angle)

        index = np.ravel_multi_index((occupancy_y, occupancy_x, dis_angle),
                                     (self.occupancy_map.info.height,
                                      self.occupancy_map.info.width,
                                      self.state_qty))
        return index

    def pose_to_index(self, pose):
        world_x = pose.pose.pose.position.x
        world_y = pose.pose.pose.position.y 
        world_yaw = self._quaternion_to_euler(pose.pose.pose.orientation)[-1]
        occupancy_x, occupancy_y = self._get_occupancy_x_y(world_x, world_y)
        index = self.occupancy_x_y_to_topological_index(occupancy_x, occupancy_y, world_yaw)
        return index

    def save_map(self, path=None, name='map', questions=[]):
        if path is None:
            os.makedirs(self.__pkg_folder+'/map/'+name, exist_ok=True)
            default_path = self.__pkg_folder+'/map/'+name+'/'+name+'.npy'                   
            np.save(default_path, self.topological_map)
            cv2.imwrite(self.__pkg_folder+'/map/'+name+'/'+name+'.jpg', self.occupancy_image)
            dict = [{'state_qty': int(self.state_qty)},
                    {'question_qty': int(self.question_qty)},
                    {'questions': questions}]
            with open(self.__pkg_folder+'/map/'+name+'/'+name+'.yaml', 'w') as file:
                yaml.dump(dict, file)
        else:
            os.makedirs(path, exist_ok=True)
            np.save(path+'/'+name+'.npy', self.topological_map)
            cv2.imwrite(path+'/'+name+'.jpg', self.occupancy_image)
            dict = [{'state_qty': int(self.state_qty)},
                    {'question_qty': int(self.question_qty)},
                    {'questions': questions}]
            with open(path+'/'+name+'.yaml', 'w') as file:
                yaml.dump(dict, file)
        return True

    def load_map(self, path='/tmp/map'):
        self.topological_map = np.load(path, allow_pickle=True)
        return True

    def index_to_pose(self, index):
        msg = Pose()
        ox, oy, angle_index = self.topological_index_to_occupancy_x_y(index)
        world_x, world_y = self._get_world_x_y(ox, oy)
        yaw = self._undiscretize_angle(angle_index)
        quaternion = self._quaternion_from_euler(0, 0, yaw)
        msg.position.x = world_x
        msg.position.y = world_y
        msg.orientation.x = quaternion[0]
        msg.orientation.y = quaternion[1]
        msg.orientation.z = quaternion[2]
        msg.orientation.w = quaternion[3]
        return msg

    def update_map(self, features=[], acc=[], glob_features=None, index=None, pose=None, cost_args=[None, None, None]):

        if index is not None:
            _index = index
        elif pose is not None:
            _index = self.pose_to_index(pose=pose)
        elif cost_args[2] is not None:
            _index = self.occupancy_x_y_to_topological_index(cost_args[0],
                                                             cost_args[1],
                                                             cost_args[2])

        self.topological_map = np.append(self.topological_map, glob_features)

        return True

    def consult_map(self, index=None, pose=None, cost_args=[None, None, None]):
        if index is not None:
            _index = index
        elif pose is not None:
            _index = self.pose_to_index(pose=pose)
        elif cost_args[1] is not None:
            _index = self.occupancy_x_y_to_topological_index(cost_args[0], cost_args[1])
        elif cost_args[2] is not None:
            _index = self.occupancy_x_y_to_topological_index(cost_args[0],
                                                             cost_args[1],
                                                             cost_args[2])   
        indexes = np.unravel_index(_index,
                                   shape=(self.occupancy_map.info.height,
                                          self.occupancy_map.info.width,
                                          self.state_qty))

        return self.topological_map[indexes[1], indexes[0], indexes[2]]

    def is_map_empty(self, index=None, pose=None, cost_args=[None, None, None]):
        if index is not None:
            _index = index
        elif pose is not None:
            _index = self.pose_to_index(pose=pose)
        elif cost_args[1] is not None:
            _index = self.occupancy_x_y_to_topological_index(cost_args[0], cost_args[1])
        elif cost_args[2] is not None:
            _index = self.occupancy_x_y_to_topological_index(cost_args[0],
                                                             cost_args[1],
                                                             cost_args[2])

        indexes = np.unravel_index(_index,
                                   shape=(self.occupancy_map.info.height,
                                          self.occupancy_map.info.width,
                                          self.state_qty))

        # not np.where(self.topological_map['index'] == _index)[0].size > 0
        # return np.all(self.topological_map[indexes[0], indexes[1], indexes[2]]) == 0
        return not np.where(self.topological_map['index'] == _index)[0].size > 0

    def _quaternion_from_euler(self, ai, aj, ak):
        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = cos(ai)
        si = sin(ai)
        cj = cos(aj)
        sj = sin(aj)
        ck = cos(ak)
        sk = sin(ak)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        q = np.empty((4, ))
        q[0] = cj*sc - sj*cs
        q[1] = cj*ss + sj*cc
        q[2] = cj*cs - sj*sc
        q[3] = cj*cc + sj*ss

        return q

    def _quaternion_to_euler(self, quaternion):
        # roll (x-axis rotation)
        sinr_cosp = 2 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
        cosr_cosp = 1 - 2 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
        roll = atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        if abs(sinp) >= 1:
            # use 90 degrees if out of range
            pitch = copysign(pi / 2, sinp)
        else:
            pitch = asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = atan2(siny_cosp, cosy_cosp)
        return [roll, pitch, yaw]

    def _occupancy_map_to_image(self):
        image = np.zeros((self.occupancy_map.info.height, self.occupancy_map.info.width),
                         dtype=np.uint8)

        for (x, y), value in np.ndenumerate(self.occupancy_matrix):
            image[x, y] = 255-int(value/100*255)
        return image

    def _get_world_x_y(self, occupancy_x, occupancy_y):
        world_x = occupancy_x * self.occupancy_map.info.resolution + \
            self.occupancy_map.info.origin.position.x
        world_y = occupancy_y * self.occupancy_map.info.resolution + \
            self.occupancy_map.info.origin.position.y
        return float(world_x), float(world_y)

    def _get_occupancy_x_y(self, world_x, world_y):
        occupancy_x = int(
            round((world_x - self.occupancy_map.info.origin.position.x)
                  / self.occupancy_map.info.resolution))
        occupancy_y = int(
            round((world_y - self.occupancy_map.info.origin.position.y)
                  / self.occupancy_map.info.resolution))
        return occupancy_x, occupancy_y

    def _get_cost_from_world_x_y(self, x, y):
        cx, cy = self._get_occupancy_x_y(x, y)
        try:
            return self._get_cost_from_occupancy_x_y(cx, cy)
        except IndexError as e:
            raise IndexError("Coordinates out of grid (in frame: {}) x: {}, y: {} must be in between: [{}, {}], [{}, {}]. Internal error: {}".format(
                self.reference_frame, x, y,
                self.occupancy_map.info.origin.position.x,
                self.occupancy_map.info.position.x + self.occupancy_map.info.height * self.occupancy_map.info.resolution,
                self.occupancy_map.info.origin.position.y,
                self.occupancy_map.info.origin.position.y + self.occupancy_map.info.width * self.occupancy_map.info.resolution,
                e))

    def _get_cost_from_occupancy_x_y(self, x, y):
        if self._is_in_occupancy(x, y):
            # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
            # first index is the row, second index the column
            return self.occupancy_matrix[y][x]
        else:
            raise IndexError(
                "Coordinates out of occupancy, x: {}, y: {} must be in between: [0, {}], [0, {}]".format(
                    x, y, self.occupancy_map.info.height, self.occupancy_map.info.width))

    def _is_in_occupancy(self, x, y):
        if -1 < x < self.occupancy_map.info.width and -1 < y < self.occupancy_map.info.height:
            return True
        else:
            return False

    def _discretize_angle(self, angle):
    
        if angle < 0:
            angle = angle + 2 * pi
        elif angle is None:
            angle = 0
        elif angle > 2 * pi:
            angle = 2 * pi

        discretized_angle = int(np.round(np.interp(angle, np.linspace(0, 2*pi, self.state_qty),
                                np.arange(self.state_qty))))

        return discretized_angle

    def _undiscretize_angle(self, discretized_angle):

        return np.interp(discretized_angle, np.arange(self.state_qty + 1),
                         np.linspace(0, 2*pi, self.state_qty + 1), left=None, right=None, period=None)
