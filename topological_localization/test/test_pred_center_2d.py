from topological_localization.main import TopologicalLocalization
from geometry_msgs.msg import PoseWithCovariance
import rclpy
import unittest
from nav_msgs.msg import OccupancyGrid
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
        self.node.map_helper = TopologicalMap(msg,8,10)



    def tearDown(self) -> None:
        """Tear down after each test method."""
        self.node.destroy_node()


    def test_centered_2d(self):

        pose = PoseWithCovariance()
        kernel_shape = (int(231),int(383))
        
        assert self.node._calculate_2d_kernel_center(pose,kernel_shape,True) == (int(231/2),int(383/2))
    def test_uncenterd(self):
      
        pose = PoseWithCovariance()
        
        kernel_shape = (int(231),int(383))
        
        assert self.node._calculate_2d_kernel_center(pose,kernel_shape,False) == (117,192)

