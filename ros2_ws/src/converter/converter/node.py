#!/opt/venv/bin/python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import pypcd4
import os
import time
import pandas as pd
import json

OUTPUT_DIRECTORY = "/home/workspace/sensor_data"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


class CameraLidarAutoRecorder(Node):
    def __init__(self):
        super().__init__("camera_lidar_auto_recorder")
        self.bridge = CvBridge()
        self.camerainfo_subs = {}
        self.detection2d_subs = {}
        self.image_subs = {}
        self.lidar_subs = {}

        self.timer = self.create_timer(5.0, self.update_subscriptions)

    def update_subscriptions(self):
        topic_list = self.get_topic_names_and_types()
        camerainfo_type = "sensor_msgs/msg/CameraInfo"
        detection2d_type = "vision_msgs/msg/Detection2DArray"
        image_type = "sensor_msgs/msg/Image"
        lidar_type = "sensor_msgs/msg/PointCloud2"

        for topic_name, types in topic_list:
            if camerainfo_type in types and topic_name not in self.detection2d_subs:
                self.get_logger().info(f"Discovered CameraInfo topic: {topic_name}")
                sub = self.create_subscription(CameraInfo, topic_name, self.camerainfo_callback_factory(topic_name), qos_profile_sensor_data)
                self.camerainfo_subs[topic_name] = sub

            if detection2d_type in types and topic_name not in self.detection2d_subs:
                self.get_logger().info(f"Discovered Detection2DArray topic: {topic_name}")
                sub = self.create_subscription(Detection2DArray, topic_name, self.detection2d_callback_factory(topic_name), qos_profile_sensor_data)
                self.detection2d_subs[topic_name] = sub

            if image_type in types and topic_name not in self.image_subs:
                self.get_logger().info(f"Discovered Image topic: {topic_name}")
                sub = self.create_subscription(Image, topic_name, self.image_callback_factory(topic_name), qos_profile_sensor_data)
                self.image_subs[topic_name] = sub

            if lidar_type in types and topic_name not in self.lidar_subs:
                self.get_logger().info(f"Discovered PointCloud2 topic: {topic_name}")
                sub = self.create_subscription(PointCloud2, topic_name, self.lidar_callback_factory(topic_name), qos_profile_sensor_data)
                self.lidar_subs[topic_name] = sub

    def camerainfo_callback_factory(self, topic: str):
        def callback(msg: CameraInfo):
            try:
                sensor_name = f"{topic.replace("/","", 1).split("/")[1].replace("_", "")}"
                output_filepath = os.path.join(OUTPUT_DIRECTORY, sensor_name, "camera_info.json")
                info = {
                    "model": msg.distortion_model,
                    "d": [d for d in msg.d],
                    "K": [k for k in msg.k],
                    # "focal_length": [msg.k[0, 0], msg.k[0, 2]],
                    # "principal_point": [msg.k[1, 1], msg.k[1, 2]],
                    "width": msg.width,
                    "height": msg.height,
                }
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                with open(output_filepath, "w") as f:
                    f.write(json.dumps(info))
                self.get_logger().info(f"Saved camera info: {output_filepath}")
                self.destroy_subscription(self.camerainfo_subs[topic])
                self.get_logger().info(f"Destroyed subscription: {topic}")

            except Exception as e:
                self.get_logger().error(f"CameraInfo conversion failed: {e}")

        return callback

    def detection2d_callback_factory(self, topic):
        def callback(msg: Detection2DArray):
            try:
                rows = []

                for detection in msg.detections:
                    ids = []
                    scores = []
                    for result in detection.results:
                        ids.append(int(result.hypothesis.class_id))
                        scores.append(int(result.hypothesis.score))
                    rows.append(
                        {
                            "id": ids[np.argmax(np.array(scores))],
                            "x": detection.bbox.center.position.x,
                            "y": detection.bbox.center.position.y,
                        }
                    )
                output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{topic.replace("/", "", 1).split("/")[1].replace("_", "")}", "detections.json")
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                pd.DataFrame(rows).to_json(output_filepath)
                self.get_logger().info(f"Saved detections: {output_filepath}")
                self.destroy_subscription(self.detection2d_subs[topic])
                self.get_logger().info(f"Destroyed subscription: {topic}")

            except Exception as e:
                self.get_logger().error(f"Detection conversion failed: {e}")

        return callback

    def image_callback_factory(self, topic):
        def callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                sensor_name = f"{topic.replace("/","", 1).split("/")[1].replace("_", "")}"
                output_filepath = os.path.join(OUTPUT_DIRECTORY, sensor_name, sensor_name + ".bmp")
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                cv2.imwrite(output_filepath, cv_image)
                self.get_logger().info(f"Saved image: {output_filepath}")
                self.destroy_subscription(self.image_subs[topic])
                self.get_logger().info(f"Destroyed subscription: {topic}")

            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")

        return callback

    def lidar_callback_factory(self, topic):
        def callback(msg):
            try:
                pc = pypcd4.PointCloud.from_msg(msg)
                sensor_name = f"{topic.replace("/", "", 1).split("/")[1].replace("_", "")}"
                output_filepath = os.path.join(OUTPUT_DIRECTORY, sensor_name, sensor_name + ".pcd")
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                pc.save(output_filepath)
                self.get_logger().info(f"Saved point cloud: {output_filepath}")
                self.destroy_subscription(self.lidar_subs[topic])
                self.get_logger().info(f"Destroyed subscription: {topic}")
            except Exception as e:
                self.get_logger().error(f"Point cloud saving failed: {e}")

        return callback


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarAutoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
