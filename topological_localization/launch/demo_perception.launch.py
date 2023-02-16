import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import yaml

def generate_launch_description():
    cv_dir = os.path.join(
      get_package_share_directory('computer_vision')
      )
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

    with open(map_config, 'r') as f:
        map_config = yaml.safe_load(f)['map_server']['ros__parameters']

    map_config['yaml_filename'] = os.path.join(cv_dir, 'maps', str(map_config['yaml_filename']))
    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('topological_localization'), 'default.rviz')]
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
        parameters=[map_config], )      

    tiago_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('computer_vision'), 'launch'),
            '/sim.launch.py'])
        )
    perception_update = Node(
         package='topological_localization',
         executable='perception_update',
         name='perception_update',
         parameters=[config, top_config],
         remappings=[('features', '/vqa/features')]
      )
    vqa = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('vqa_ros')),
            '/vqa.launch.py'])
        )

    return LaunchDescription([
        map,
        map_server,
        vqa,
        rviz,
        # visualizer,
        # tiago_sim,        
        perception_update        
   ])