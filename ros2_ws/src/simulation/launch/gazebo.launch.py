import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, Command

def generate_launch_description():
    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name = 'simulation'  # <--- CHANGE ME

    base_path = Path(get_package_share_directory(package_name))
    world_path = base_path / 'worlds'
    xacro_file = os.path.join(base_path, 'urdf', 'robot.urdf')

    # Include the Gazebo launch file, provided by the gazebo_ros package

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments=[('gz_args', [f" -r -v 2 {world_path / 'empty.sdf'}"])]
    )
    
    path_to_urdf = os.path.join(get_package_share_directory(package_name), 'urdf', 'robot.urdf.xacro')
    
    # Create a robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', str(path_to_urdf)]), value_type=str)
        }]
    )
    # Spawn robot directly from Xacro
    spawn_entity = Node(package='ros_gz_sim', executable='create',
                                arguments=[
                                    '-file', xacro_file,
                                    '-name', 'vehicle',
                                    '-x', '0.0',
                                    '-y', '0.0',
                                    '-z', '0.0',
                                ],
                                output='screen'
                                )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "robot1",
            "-topic",
            "/robot_description",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "1.4",
        ],
        output="screen",
    )

    spawn = ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', 
                 '-file', xacro_file, '-entity', 'my_robot'],
            output='screen'
        )
    # Launch them all!
    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        #spawn_entity
        TimerAction(
            period=5.0,  # Delay in seconds
            actions=[
                spawn_entity
            ]
        ),
    ])
    