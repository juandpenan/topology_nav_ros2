from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
import pickle
import pathlib
from copy import deepcopy

"""
Basic navigation demo to follow a given path after smoothing
"""

def save_path_to_disc():
    navigator = BasicNavigator()

    msg_path = str(pathlib.Path(__file__).parent.resolve()).removesuffix('/topological_localization') + '/resource/path_msg.pkl'


    # Set our demo's initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0

    initial_pose.pose.orientation.z = 0.0
    initial_pose.pose.orientation.w = 1.0
    navigator.setInitialPose(initial_pose)


    # define poses [x, y, z, w]

    experiment_route = [
        [1.685354400687667, 0.08015064901073686, 0.030892746274249386, 0.9995227052086585],
        [1.68471894838796, 2.9597302777869507, 0.654768205660711, 0.7558297406531796],
        [11.082962975589972, 3.8365504459400444, 0.21160636869221627, 0.9773549737576894],
        [12.51810787810021, 5.563021797062294, 0.4326437380674422, 0.901564970432542],
        [12.303027677227687, 9.120557881291557, 0.3747192625880009, 0.927138325302921]
        ]
    
    goals = []
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    for pt in experiment_route:
        goal_pose.pose.position.x = pt[0]
        goal_pose.pose.position.y = pt[1]
        goal_pose.pose.orientation.z = pt[2]
        goal_pose.pose.orientation.w = pt[3]
        goals.append(deepcopy(goal_pose))

        



    navigator.followWaypoints(goals)



def main():
    rclpy.init()

    save_path_to_disc()

  


if __name__ == '__main__':
    main()