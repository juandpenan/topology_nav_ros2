import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    map_config = os.path.join(
      get_package_share_directory('topological_mapping'),
      'params.yaml'
      )

    config = os.path.join(
      get_package_share_directory('topological_localization'), 
      'params.yaml'
      )
    top_config = os.path.join(
        get_package_share_directory('topological_mapping'),
        'topomap_params.yaml'
    )
    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('topological_localization'), 'default.rviz')]
        )
    map = Node(
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
        parameters=[map_config])

    msg_saver = Node(
         package='topological_localization',
         executable='msg_saver',
         name='msg_saver',
         parameters=[config, top_config]
      )

    return LaunchDescription([
        map,
        map_server,
        rviz,
        msg_saver])
