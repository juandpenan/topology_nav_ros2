import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import yaml


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
    cv_dir = os.path.join(
      get_package_share_directory('computer_vision')
      )
    with open(map_config, 'r') as f:
        map_config = yaml.safe_load(f)['map_server']['ros__parameters']

    map_config['yaml_filename'] = os.path.join(cv_dir, 'maps', str(map_config['yaml_filename']))

    perception_update = Node(
         package='topological_localization',
         executable='perception_update',
         name='perception_update',
         parameters=[config, top_config],
         remappings=[('features', '/vqa/features')]
      )
    topo_localization = Node(
         package='topological_localization',
         executable='localization',
         name='localization',
         parameters=[config, top_config],
         remappings=[
            ('odom','/ground_truth_odom'),
            ('features','/vqa/features')
         ]
      )
    vqa = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('vqa_ros')),
            '/vqa.launch.py'])
        )
    mh_amcl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('mh_amcl'), 'launch', 'tiago_launch.py')
            ])
        )

    return LaunchDescription([
        mh_amcl,
        vqa,
        # topo_localization
        perception_update,
   ])