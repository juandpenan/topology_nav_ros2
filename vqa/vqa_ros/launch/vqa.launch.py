import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node

import yaml


ros2_serial_example_directory = os.path.join(
    get_package_share_directory('vqa_ros'))

param_config = os.path.join(ros2_serial_example_directory,
                            'params.yaml')

with open(param_config, 'r') as f:
    params = yaml.safe_load(f)['vqa_node']['ros__parameters']

print(params)


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vqa_ros',
            executable='vqa',
            name='vqa',
            output='screen',
            emulate_tty=True,
            parameters=[params],
            remappings=[
                ('image', '/xtion/rgb/image_raw'),
                # ('image', '/head_front_camera/rgb/image_raw'),
                ('camera_info', '/camera/rgb/camera_info'),
            ],
        )
    ])
