#!/opt/venv/bin/python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import pypcd4
import os
import time

OUTPUT_DIR = "/home/workspace/sensor_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CameraLidarAutoRecorder(Node):
    def __init__(self):
        super().__init__('camera_lidar_auto_recorder')
        self.bridge = CvBridge()
        self.image_subs = {}
        self.lidar_subs = {}

        self.timer = self.create_timer(5.0, self.update_subscriptions)

    def update_subscriptions(self):
        topic_list = self.get_topic_names_and_types()
        image_type = 'sensor_msgs/msg/Image'
        lidar_type = 'sensor_msgs/msg/PointCloud2'

        for topic_name, types in topic_list:
            if image_type in types and topic_name not in self.image_subs:
                self.get_logger().info(f"Discovered image topic: {topic_name}")
                sub = self.create_subscription(
                    Image, topic_name, self.image_callback_factory(topic_name),
                    qos_profile_sensor_data
                )
                self.image_subs[topic_name] = sub

            if lidar_type in types and topic_name not in self.lidar_subs:
                self.get_logger().info(f"Discovered lidar topic: {topic_name}")
                sub = self.create_subscription(
                    PointCloud2, topic_name, self.lidar_callback_factory(topic_name),
                    qos_profile_sensor_data
                )
                self.lidar_subs[topic_name] = sub

    def image_callback_factory(self, topic):
        def callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                filename = f"{topic.replace('/', '_')}_{int(time.time()*1000)}.bmp"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, cv_image)
                self.get_logger().info(f"Saved image: {filepath}")
            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")
        return callback

    def lidar_callback_factory(self, topic):
        def callback(msg):
            try:
                pc = pypcd4.PointCloud.from_msg(msg)
                filename = f"{topic.replace('/', '_')}_{int(time.time()*1000)}.pcd"
                filepath = os.path.join(OUTPUT_DIR, filename)
                pc.save(filepath)
                self.get_logger().info(f"Saved point cloud: {filepath}")
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

if __name__ == '__main__':
    main()
