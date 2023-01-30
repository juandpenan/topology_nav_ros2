from setuptools import setup
import os
from glob import glob

package_name = 'topological_mapping'
topological_map = "topological_mapping/topological_mapping"

setup(
    name=package_name,
    version='0.0.0',
    # packages=[package_name,topological_map],
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('config/*')),
        (os.path.join('share', package_name), glob('map/**/*.npy')),
        (os.path.join('share', package_name), glob('map/**/*.jpg'))
        # (os.path.join('share', package_name), glob('map/map2/*.npy')),
        # (os.path.join('share', package_name), glob('map/map3/*.npy')),
        # (os.path.join('share', package_name), glob('map/map4/*.jpg'))
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
            'mapping_node = topological_mapping.main:main',
            'generate_map = topological_mapping.topological_mapping.images_to_map:main',
            'show_generated_map = topological_mapping.show_mapped_coordinates:main'     
        ],
    },
)
