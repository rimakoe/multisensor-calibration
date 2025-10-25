import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, Command
from launch.event_handlers import OnProcessExit

WORLD = "boxes_circle_noise.urdf.xacro"
ROBOT = "vehicle_only_camera.urdf.xacro"

camera_names = ["front_camera_wide", "top_view_camera_front", "top_view_camera_rear", "top_view_camera_left", "top_view_camera_right"]
lidar_names = ["ref_lidar", "ref_lidar_front", "ref_lidar_right", "ref_lidar_rear", "ref_lidar_left"]


def create_lidar_bridges(lidar_topic_namepsace: str = "lidar") -> list[str]:
    lidar_bridges = []
    for name in lidar_names:
        prefix = "/" + "/".join([lidar_topic_namepsace, name])
        topic_points = "/".join([prefix, "points"])
        lidar_bridges.append("@".join([topic_points, "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"]))
    return lidar_bridges


def create_camera_bridges(camera_topic_namepsace: str = "camera") -> list[str]:
    camera_bridges = []
    for name in camera_names:
        prefix = "/" + "/".join([camera_topic_namepsace, name])
        topic_info = "/".join([prefix, "camera_info"])
        topic_image = "/".join([prefix, "bbox_image"])
        topic_bbox = "/".join([prefix, "bbox"])
        camera_bridges.append("@".join([topic_info, "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"]))
        camera_bridges.append("@".join([topic_image, "sensor_msgs/msg/Image", "gz.msgs.Image"]))
        camera_bridges.append("@".join([topic_bbox, "vision_msgs/msg/Detection2DArray", "gz.msgs.AnnotatedAxisAligned2DBox_V"]))
    return camera_bridges


def generate_launch_description():
    package_name = "simulation"
    base_path = Path(get_package_share_directory(package_name))
    path_to_robot_xacro = os.path.join(base_path, "robots", ROBOT)
    path_to_world_xacro = os.path.join(base_path, "worlds", WORLD)
    path_to_world_sdf = path_to_world_xacro.strip(".xacro").replace(".urdf", ".sdf")

    launch_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                )
            ]
        ),
        launch_arguments=[("gz_args", [f" -r -v 2 {path_to_world_sdf}"])],
    )

    # Generate the world sdf from its xacro
    generate_sdf_from_xacro = ExecuteProcess(
        cmd=["ros2", "run", "xacro", "xacro", path_to_world_xacro, ">", path_to_world_xacro.strip(".xacro").replace(".urdf", ".sdf")],
        output="screen",
        shell=True,
        on_exit=LogInfo(msg="Successfully generated world sdf file from xacro. Starting Gazebo..."),
    )

    # Ensure Gazebo starts only after the Xacro conversion is done
    launch_gazebo_after_xacro = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=generate_sdf_from_xacro,
            on_exit=[launch_gazebo],
        )
    )

    # Create a robot_state_publisher node
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": ParameterValue(Command(["xacro ", str(path_to_robot_xacro)]), value_type=str)}],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "vehicle", "-topic", "/robot_description", "-x", "0", "-y", "0", "-z", "0.0"],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ]
        + create_camera_bridges()
        + create_lidar_bridges(),
        output="log",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(base_path, "config", "standard.rviz")],
    )

    # Launch them all!
    return LaunchDescription(
        [
            generate_sdf_from_xacro,
            launch_gazebo_after_xacro,
            bridge,
            node_robot_state_publisher,
            TimerAction(period=5.0, actions=[rviz2]),  # Delay in seconds
            TimerAction(period=5.0, actions=[spawn_entity]),  # Delay in seconds
        ]
    )
