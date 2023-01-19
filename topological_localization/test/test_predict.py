from topological_localization.main import TopologicalLocalization
from geometry_msgs.msg import PoseWithCovariance
import rclpy
import unittest
from nav_msgs.msg import OccupancyGrid, Odometry
from vqa_msgs.msg import VisualFeatures
import numpy as np
from topological_mapping.topological_mapping.topological_map import TopologicalMap

class PredStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Set up before first test method."""
        # Initialize the ROS context for the test node
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down after last test method."""
        # Shutdown the ROS context
        rclpy.shutdown()

    def setUp(self) -> None:
        """Set up before each test method."""

        self.node = TopologicalLocalization()
        msg = OccupancyGrid()
        msg.info.height = 231
        msg.info.width = 383
        msg.info.origin.position.x = -9.62
        msg.info.origin.position.y = -5.83
        msg.info.resolution = 0.05
        
        self.node.map_helper = TopologicalMap(msg,8,10)
        



    def tearDown(self) -> None:
        """Tear down after each test method."""
        self.node.destroy_node()


    def test_centered_2d(self):

        pose = PoseWithCovariance()
        kernel_shape = (int(231),int(383))
        
        assert self.node._calculate_2d_kernel_center(pose,kernel_shape,True) == (int(231/2),int(383/2))

    def test_uncenterd_2d(self):
      
        pose = PoseWithCovariance()
        
        kernel_shape = (int(231),int(383))
        
        assert self.node._calculate_2d_kernel_center(pose,kernel_shape,False) == (117,192)

        kernel_shape = (int(50),int(50))

        pose.pose.position.x = -9.62
        pose.pose.position.y = -5.83
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.assertEqual(self.node._calculate_2d_kernel_center(pose, kernel_shape, False), (0, 0))

        pose.pose.position.x = -8.62
        pose.pose.position.y = -5.83
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.assertEqual(self.node._calculate_2d_kernel_center(pose, kernel_shape, False), (0, 3))

        pose.pose.position.x = -7.62
        pose.pose.position.y = -5.83
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.assertEqual(self.node._calculate_2d_kernel_center(pose, kernel_shape, False), (0, 7))

        pose.pose.position.x = -7.62
        pose.pose.position.y = -4.83
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.assertEqual(self.node._calculate_2d_kernel_center(pose, kernel_shape, False), (3, 7))
    def test_center_1d(self):
        pose = PoseWithCovariance()
        kernel_shape = 100

        self.assertEqual(self.node._calculate_1d_kernel_center(pose, kernel_shape, True), 50)

        self.assertEqual(self.node._calculate_1d_kernel_center(pose, kernel_shape, False), 0)

        q = self.node.map_helper._quaternion_from_euler(0.0, 0.0, 2 * np.pi)

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.assertEqual(self.node._calculate_1d_kernel_center(pose, kernel_shape, False), 100)

        q = self.node.map_helper._quaternion_from_euler(0.0, 0.0, np.pi)

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.assertEqual(self.node._calculate_1d_kernel_center(pose, kernel_shape, False), 50)

        q = self.node.map_helper._quaternion_from_euler(0.0, 0.0, np.pi/2)

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.assertEqual(self.node._calculate_1d_kernel_center(pose, kernel_shape, False), 25)

    def test_2d_gaussian(self):

        kernel = self.node._2d_gaussian_kernel()
        self.assertEqual(kernel.shape, (10, 10))
        self.assertAlmostEqual(np.sum(kernel), 1.0, delta=1e-6)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2, 2))

        # Test with custom kernel size
        kernel = self.node._2d_gaussian_kernel(k_size=(5, 5))
        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(np.sum(kernel), 1.0, delta=1e-6)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2, 2))

        # Test with custom standard deviation
        kernel = self.node._2d_gaussian_kernel(sig=[2, 2])
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2, 2))

        # Test with custom center
        kernel = self.node._2d_gaussian_kernel(center=(3, 3))
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (3, 3))

        with self.assertRaises(ZeroDivisionError):
            self.node._2d_gaussian_kernel(sig=[0, 0])

    def test_1d_gaussian(self):

        kernel = self.node._1d_gaussian_kernel()
        self.assertEqual(kernel.shape, (5,))
        self.assertAlmostEqual(np.sum(kernel), 1.0, delta=1e-6)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2,))

        # Test with custom kernel size
        kernel = self.node._1d_gaussian_kernel(k_size=10)
        self.assertEqual(kernel.shape, (10,))
        self.assertAlmostEqual(np.sum(kernel), 1.0, delta=1e-6)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2,))

        # Test with custom standard deviation
        kernel = self.node._1d_gaussian_kernel(sigma=2)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (2,))

        # Test with custom center
        kernel = self.node._1d_gaussian_kernel(center=3)
        self.assertEqual(np.unravel_index(np.argmax(kernel, axis=None), kernel.shape), (3,))

    def test_predict(self):

        self.node.init_localization_grid()

        # set a max
        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,1] = 10000000000000000000000000000000.0

        # test max
        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value1 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertEqual(ind, (117, 192))
        self.node.prediction_step(0, 0)

        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value2 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertGreater(value1, value2)


        self.node.odom_pose = Odometry().pose
        self.node.odom_pose.pose.position.x = 2.0
        self.node.odom_pose.pose.position.y = 0.0
        self.node.odom_pose.pose.position.z = 0.0
        self.node.odom_pose.pose.orientation.w = 1.0

        self.node.init_localization_grid()

        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,1] = 10000000000000000000000000000000.0

        self.node.prediction_step(0.3, 0)

         # test max
        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value1 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertEqual(ind, (118, 213))

        self.node.odom_pose = Odometry().pose
        self.node.odom_pose.pose.position.x = 0.0
        self.node.odom_pose.pose.position.y = -2.0
        self.node.odom_pose.pose.position.z = 0.0
        self.node.odom_pose.pose.orientation.w = 1.0

        self.node.init_localization_grid()

        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,1] = 10000000000000000000000000000000.0

        self.node.prediction_step(0.3, 0)

         # test max
        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value1 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertEqual(ind, (98, 193))

        self.node.odom_pose = Odometry().pose
        self.node.odom_pose.pose.position.x = 0.0
        self.node.odom_pose.pose.position.y = 2.0
        self.node.odom_pose.pose.position.z = 0.0
        self.node.odom_pose.pose.orientation.w = 1.0

        self.node.init_localization_grid()

        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,1] = 10000000000000000000000000000000.0

        self.node.prediction_step(0.3, 0)

         # test max
        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value1 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertEqual(ind, (138, 193))

        self.node.odom_pose = Odometry().pose
        self.node.odom_pose.pose.position.x = -1.0
        self.node.odom_pose.pose.position.y = 2.0
        self.node.odom_pose.pose.position.z = 0.0
        self.node.odom_pose.pose.orientation.w = 1.0

        self.node.init_localization_grid()

        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,1] = 10000000000000000000000000000000.0

        self.node.prediction_step(0.3, 0)

         # test max
        ind = np.unravel_index(np.argmax(self.node._localization_grid[:, :, 0], axis=None), self.node._localization_grid[:,:,0].shape)
        value1 = self.node._localization_grid[ind[0], ind[1], 0]

        self.assertEqual(ind, (138, 183))

        self.node.odom_pose = Odometry().pose
        self.node.odom_pose.pose.position.x = -1.0
        self.node.odom_pose.pose.position.y = 2.0
        self.node.odom_pose.pose.position.z = 0.0

        q = self.node.map_helper._quaternion_from_euler(0.0, 0.0, 0.0)

        self.node.odom_pose.pose.orientation.x = q[0]
        self.node.odom_pose.pose.orientation.y = q[1]
        self.node.odom_pose.pose.orientation.z = q[2]
        self.node.odom_pose.pose.orientation.w = q[3]
      

        self.node.init_localization_grid()

        self.node._localization_grid[117,192,0] = 1000000000000000000000000000000000000000000000000.0
        self.node._localization_grid[117,192,3] = 10000000000000000000000000000000.0
        print(self.node._localization_grid[117, 192, 1:])
        self.node.prediction_step(0.3, 0.1)

         # test max
        ind_x_y = np.unravel_index(np.argmax(self.node._localization_grid, axis=None), self.node._localization_grid.shape)
        ind_angle = np.unravel_index(np.argmax(self.node._localization_grid[ind[0], ind[1], 1:], axis=None), self.node._localization_grid[ind[0], ind[1], 1:].shape)
        print(self.node._localization_grid[117, 192, 1:])

        self.assertEqual(ind_angle, (1,))

    def test_perception(self):
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
        features = np.array(features)

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
 
        acc = np.array(acc)

        current_features = ['black',
                            'black',
                            'black',
                            'black',
                            'black',
                            'black',
                            'black',
                            'black',
                            'black',
                            'black']
        current_acc = [0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8,
                       0.8]
                       

        # pose = Odometry()
        # pose.pose.pose.position.x = 0.0
        # pose.pose.pose.position.y = 0.0
        # pose.pose.pose.position.z = 0.0

        # q = self.node.map_helper._quaternion_from_euler(0.0, 0.0, 0.0)

        # pose.pose.pose.orientation.x = q[0]
        # pose.pose.pose.orientation.y = q[1]
        # pose.pose.pose.orientation.z = q[2]
        # pose.pose.pose.orientation.w = q[3]
        self.node.vqa_features = VisualFeatures()
        self.node.vqa_features.data = current_features
        self.node.vqa_features.acc = current_acc
        self.node.map_helper.update_map(features=features, acc=acc, cost_args=[1,3,np.pi])
        self.node.init_localization_grid()
        self.node.perception_update()

        ind = np.unravel_index(np.argmax(self.node._localization_grid, axis=None), self.node._localization_grid.shape)
        #change this but works
        self.assertEqual(ind,(1,3,0))


        











