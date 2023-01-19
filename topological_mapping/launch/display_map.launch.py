import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions import LifecycleNode
from launch_ros.actions import SetRemap


def generate_launch_description():

    config = os.path.join(
      get_package_share_directory('topological_mapping'),
      'params.yaml'
      )
    top_config = os.path.join(
        get_package_share_directory('topological_mapping'),
        'topomap_params.yaml'
    )
    map =  Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                output='screen',
                parameters=[{'use_sim_time': True},
                            {'autostart': True},
                            {'node_names': ["map_server"]}])

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        respawn=True,
        respawn_delay=2.0,
        parameters=[config], )

    show_map_node = Node(
        package='topological_mapping',
        executable='show_generated_map',
        name='mapper_display',
        parameters=[top_config]
   
        # respawn=True,
        # respawn_delay=2.0

    )      
   
    return LaunchDescription([
        map_server,  
        map,
        # mh_amcl,
        # map_vis
        show_map_node,         
   ])