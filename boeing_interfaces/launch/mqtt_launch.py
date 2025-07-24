from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import XMLLaunchDescriptionSource  # Use this for XML files
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Define the path to the params file
    params_file = LaunchConfiguration("params_file", default="/home/students/girgine/ros2_ws/src/boeing_interfaces/config/mqtt_params.yaml")

    # Path to the standalone.launch.ros2.xml file in the mqtt_client package
    mqtt_client_launch_path = PathJoinSubstitution([
        FindPackageShare("mqtt_client"),
        "launch",
        "standalone.launch.ros2.xml"
    ])

    # Include the standalone launch file and pass the params_file argument
    mqtt_client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mqtt_client_launch_path),
        launch_arguments={
            "params_file": params_file,
        }.items()
    )

    return LaunchDescription([
        mqtt_client_launch,
    ])
