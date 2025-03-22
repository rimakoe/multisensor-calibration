import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource



def generate_launch_description():
    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name = 'simulation'  # <--- CHANGE ME

    base_path = Path(get_package_share_directory(package_name))
    world_path = base_path / 'worlds'

    # Include the Gazebo launch file, provided by the gazebo_ros package

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments=[('gz_args', [f" -r -v 2 {world_path / 'empty.sdf'}"])]
    )
    # Launch them all!
    return LaunchDescription([
        gazebo
    ])