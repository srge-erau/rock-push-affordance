from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'boeing_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='girgine',
    maintainer_email='girgine@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localizer = boeing_vision.orientation:main',
            'depth_image = boeing_vision.depth_image:main',
            'normal_estimator = boeing_vision.normal_estimator:main',
            'pointcloud_segmentation = boeing_vision.pointcloud_segmentation:main',
            'obstacle_feature_extractor = boeing_vision.obstacle_feature_extractor:main',
            'filter_pointcloud = boeing_vision.filter_pointcloud:main',
            'bag_player = boeing_vision.bag_player:main'
        ],
    },
)
