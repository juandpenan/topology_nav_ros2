import copy
import unittest

from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid

import numpy as np

from topological_mapping.topological_mapping.topological_map import TopologicalMap


class PredStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Set up before first test method."""
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down after last test method."""
        pass

    def setUp(self) -> None:
        """Set up before each test method."""
        self.msg = OccupancyGrid()
        self.msg.info.height = 231
        self.msg.info.width = 383
        self.msg.info.resolution = 0.05
        self.msg.info.origin.position.x = -9.62
        self.msg.info.origin.position.y = -5.83
        self.topo_map = TopologicalMap(self.msg, 8, 10)
        self.topo_map.occupancy_image = np.zeros([100, 100, 3], dtype=np.uint8)

    def tearDown(self) -> None:
        """Tear down after each test method."""
        del self.topo_map

    def test_index_to_occupancy(self):
        x, y, z = self.topo_map.topological_index_to_occupancy_x_y((231*383*8)-1)
        assert x == 382 and y == 230 and z == 7

    def test_occupancy_to_index(self):
        index = self.topo_map.occupancy_x_y_to_topological_index(0, 0, 0.3)

        assert index == 0

    def test_both_directions(self):
        # Height (y / rows):
        # Width (x / columns)
        x = np.random.randint(0, 383)
        y = np.random.randint(0, 231)
        z = np.random.uniform(0, 2*np.pi)

        index = self.topo_map.occupancy_x_y_to_topological_index(x, y, z)

        # z = self.topo_map._discretize_angle(z)
        z = self.topo_map._discretize_angle(z)

        x_i, y_i, z_i = self.topo_map.topological_index_to_occupancy_x_y(index)

        assert x == x_i and y == y_i and z == z_i

    def test_both_directions_2(self):

        index = np.random.randint(0, (231*383*8) - 1)

        x, y, z = self.topo_map.topological_index_to_occupancy_x_y(index)

        index_2 = self.topo_map.occupancy_x_y_to_topological_index(x, y, np.interp(z, np.arange(8), np.linspace(0,2*np.pi,8), left=None, right=None, period=None))

        delta = 0.0001
        # error message in case if test case got failed
        message = 'first and second are not almost equal.'
        # assert function() to check if values are almost equal
        self.assertAlmostEqual(index, index_2, None, message, delta)

    def test_index_to_occupancy_2(self):
        x, y, z = self.topo_map.topological_index_to_occupancy_x_y(118516)
        assert x == 260 and y == 38 and z == 4

    def test_pose_to_index(self):

        pos = PoseWithCovarianceStamped()
        pos.pose.pose.position.x = -9.62
        pos.pose.pose.position.y = -5.83
        pos.pose.pose.position.z = 0.0
        pos.pose.pose.orientation.x = 0.0
        pos.pose.pose.orientation.y = 0.0
        pos.pose.pose.orientation.z = 0.0
        pos.pose.pose.orientation.w = 1.0

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(0, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 1.9*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(2, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 0.9*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(1, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 2*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(2, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 2.1*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(2, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 2.9*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(3, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 3.01*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(3, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 3.99*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(3, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 2*np.pi)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(7, index)

        q = self.topo_map._quaternion_from_euler(0, 0, 8*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(7, index)

        q = self.topo_map._quaternion_from_euler(0, 0, -0.5*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(7, index)

        q = self.topo_map._quaternion_from_euler(0, 0, -0.6*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(6, index)

        q = self.topo_map._quaternion_from_euler(0, 0, -1.9*2*np.pi/8)

        pos.pose.pose.orientation.x = q[0]
        pos.pose.pose.orientation.y = q[1]
        pos.pose.pose.orientation.z = q[2]
        pos.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(5, index)

        pos.pose.pose.position.x = -9.62 + 0.051
        pos.pose.pose.position.y = -5.83
        pos.pose.pose.orientation.x = 0.0
        pos.pose.pose.orientation.y = 0.0
        pos.pose.pose.orientation.z = 0.0
        pos.pose.pose.orientation.w = 1.0

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(8, index)

        pos.pose.pose.position.x = -9.62 + 0.051*2
        pos.pose.pose.position.y = -5.83
        pos.pose.pose.orientation.x = 0.0
        pos.pose.pose.orientation.y = 0.0
        pos.pose.pose.orientation.z = 0.0
        pos.pose.pose.orientation.w = 1.0

        index = self.topo_map.pose_to_index(pos)

        self.assertEqual(16, index)

    def test_index_to_pose(self):

        index = 0

        pose = self.topo_map.index_to_pose(index)

        self.assertEqual(first=[pose.position.x,
                                pose.position.y,
                                pose.position.z,
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w],
                         second=[-9.62,
                                 -5.83,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 1.0])

    def test_both_directions_index_and_pose(self):

        index = np.random.randint(0, (231*383*8) - 1)

        pose = self.topo_map.index_to_pose(index)

        pos_msg = PoseWithCovarianceStamped()
        pos_msg.pose.pose = copy.deepcopy(pose)
        index_2 = self.topo_map.pose_to_index(pos_msg)

        self.assertEqual(index, index_2)

        pos_msg.pose.pose.position.x = np.random.uniform(0,
                                                         (self.topo_map.occupancy_map.info.width *
                                                          self.topo_map.occupancy_map.info.resolution)/2)

        pos_msg.pose.pose.position.y = np.random.uniform(0,
                                                         (self.topo_map.occupancy_map.info.height *
                                                          self.topo_map.occupancy_map.info.resolution)/2)
        pos_msg.pose.pose.position.z = 0.0

        q = self.topo_map._quaternion_from_euler(0, 0, np.random.uniform(0, 2*np.pi))

        pos_msg.pose.pose.orientation.x = q[0]
        pos_msg.pose.pose.orientation.y = q[1]
        pos_msg.pose.pose.orientation.z = q[2]
        pos_msg.pose.pose.orientation.w = q[3]

        index = self.topo_map.pose_to_index(pos_msg)

        pos_msg2 = self.topo_map.index_to_pose(index)

        self.assertAlmostEqual(first=sum([pos_msg.pose.pose.position.x,
                                          pos_msg.pose.pose.position.y,
                                          pos_msg.pose.pose.position.z,
                                          pos_msg.pose.pose.orientation.x,
                                          pos_msg.pose.pose.orientation.y,
                                          pos_msg.pose.pose.orientation.z,
                                          pos_msg.pose.pose.orientation.w]),
                               second=sum([pos_msg2.position.x,
                                           pos_msg2.position.y,
                                           pos_msg2.position.z,
                                           pos_msg2.orientation.x,
                                           pos_msg2.orientation.y,
                                           pos_msg2.orientation.z,
                                           pos_msg2.orientation.w]),
                               delta=0.05*3)

    def test_update_map(self):
        features = [['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black'],
                    ['black', 'black']]
        acc = [[0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8],
               [0.8, 0.8]]
 
        indexes = []

        for x in range(50):
            pos_msg = PoseWithCovarianceStamped()
            pos_msg.pose.pose.position.x = np.random.uniform(0,
                                                             (self.topo_map.occupancy_map.info.width *
                                                              self.topo_map.occupancy_map.info.resolution)/3)

            pos_msg.pose.pose.position.y = np.random.uniform(0,
                                                             (self.topo_map.occupancy_map.info.height *
                                                              self.topo_map.occupancy_map.info.resolution)/3)
            pos_msg.pose.pose.position.z = 0.0

            q = self.topo_map._quaternion_from_euler(0, 0, np.random.uniform(0, 2*np.pi))

            pos_msg.pose.pose.orientation.x = q[0]
            pos_msg.pose.pose.orientation.y = q[1]
            pos_msg.pose.pose.orientation.z = q[2]
            pos_msg.pose.pose.orientation.w = q[3]
      
            indexes.append(self.topo_map.pose_to_index(pos_msg))
            
            del pos_msg

        for x in range(len(indexes)):
            self.topo_map.update_map(features, acc, indexes[x])

        self.topo_map.save_map('/tmp')

        test_map = np.load('/tmp/map.npy', allow_pickle=True)
        sum = 0
        for row in range(test_map.shape[0]):
            for colum in range(test_map.shape[1]):
                for state in range(test_map.shape[2]):
                    for i in range(10):
                        if test_map[row, colum, state, i][0] != 0:
                            sum += 1

        self.assertAlmostEqual(sum/10, 50)
