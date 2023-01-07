from rclpy.node import Node
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped


class TestNode(Node):
    def __init__(self):
        super().__init__('node_test')
        self.create_subscription(PoseWithCovarianceStamped,"odom",self.callback,1)

    def callback(self,msg):
        print(msg)




def main(args=None):
    rclpy.init(args=args)  
    node = TestNode()
    rclpy.spin(node)


    node.destroy_node()
    rclpy.shutdown()

