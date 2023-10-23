import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

import yaml


def generate_launch_description():

    config = os.path.join(
      get_package_share_directory('topological_mapping'),
      'params.yaml'
      )
    cv_dir = get_package_share_directory('computer_vision')

    ros2_serial_example_directory = os.path.join(get_package_share_directory('vqa_ros'))

    param_config = os.path.join(ros2_serial_example_directory,
                                'params.yaml')

    with open(param_config, 'r') as f:
        params = yaml.safe_load(f)['vqa_node']['ros__parameters']
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)['map_server']['ros__parameters']
        
        conf['yaml_filename'] = os.path.join(cv_dir,
                                             'maps',
                                             str(conf['yaml_filename']))
        config = conf

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
        parameters=[config], )

    save_disc = Node(
        package='topological_mapping',
        executable='generate_map',
        name='save_to_disc',
        parameters=[params]
    )
    return LaunchDescription([
        map_server,
        map,
        save_disc
    ])