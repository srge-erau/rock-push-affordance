import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the other package's launch file
    other_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )  

    return LaunchDescription([
        # Include the other package's launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(other_launch_file),
            launch_arguments = {
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                #'enable_color': 'true'
            }.items()
        ),
        
        # Node 1 from this package
        #Node(
        #    package='boeing_vision',
        #    executable='localizer',
        #    name='my_localizer'
        #),
        
        # Node 2 from this package
        #Node(
        #    package='boeing_vision',
        #    executable='normal_estimator',
        #    name='my_normal_estimator'
        #)
    ])