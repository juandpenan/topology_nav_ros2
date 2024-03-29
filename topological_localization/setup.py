from setuptools import setup
import os
from glob import glob

package_name = 'topological_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juan',
    maintainer_email='juan97pena@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization = topological_localization.main:main',
            'motion_update = topological_localization.motion_update:main',
            'perception_update = topological_localization.perception_update:main',
            'msg_saver = topological_localization.save_marker_to_disk:main',
            'experiment1 = topological_localization.experiment1:main'
        ],
    },
)
