#!/usr/bin/env python
import ctypes
import struct
import time
import json
import os
import open3d as o3d
from datetime import datetime

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import rclpy.time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs_py import point_cloud2

class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")
        # Set frequency
        self.rate = self.create_rate(1)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(
            self, Image, "/camera/color/image_raw"
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, "/camera/depth/image_raw"
        )
        self.camera_info_sub = message_filters.Subscriber(
            self, CameraInfo, "/camera/color/camera_info"
        )
        self.point_cloud_sub = message_filters.Subscriber(
            self, PointCloud2, "/camera/depth_registered/points"
        )

        # Create TF Listener to get the transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # self.image_sub = self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        # self.depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        # self.camera_info_sub = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 10)

        ts = message_filters.TimeSynchronizer(
            [self.image_sub, self.depth_sub, self.camera_info_sub, self.point_cloud_sub], 10
        )
        ts.registerCallback(self.callback)
        self.image_count = 0
        now = datetime.now()
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.save_dir = f"{cwd}/../demo_data/images_{now.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, "rgb"))
            os.makedirs(os.path.join(self.save_dir, "depth"))
            os.makedirs(os.path.join(self.save_dir, "camera_poses"))
            os.makedirs(os.path.join(self.save_dir, "point_cloud"))

    def callback(self, image, depth, camera_info, point_cloud):
        self.image_callback(image)
        self.depth_callback(depth)
        self.camera_info_callback(camera_info)
        # self.point_cloud_callback(point_cloud)
        self.image_count += 1

    def image_callback(self, data):
        try:
            self.frame_id = data.header.frame_id
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_path = os.path.join(
                self.save_dir, "rgb", f"image_{self.image_count:04d}.png"
            )
            cv2.imwrite(image_path, cv_image)
            self.get_logger().info(f"Saved image {self.image_count}")
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

    def depth_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            image_path = os.path.join(
                self.save_dir, "depth", f"image_{self.image_count:04d}.png"
            )
            cv2.imwrite(image_path, cv_image)
            self.get_logger().info(f"Saved depth {self.image_count}")
        except Exception as e:
            self.get_logger().error(f"Failed to save depth: {e}")

    def camera_info_callback(self, data):
        if self.image_count == 0:
            # Create dictionary with camera info
            camera_info = {
                "width": data.width,
                "height": data.height,
                "intrinsic_matrix": data.k.tolist(),
                "D": data.d.tolist(),
                "R": data.r.tolist(),
                "P": data.p.tolist(),
            }
            with open(
                os.path.join(self.save_dir, "camera_intrinsics.json"), "w"
            ) as file:
                json.dump(camera_info, file)
        try:
            from_frame_rel = "base_link"
            to_frame_rel = self.frame_id  # "orbbec_base"
            t_base_link_2_camera = self.tf_buffer.lookup_transform(
                from_frame_rel, to_frame_rel, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return
        rot_matrix = R.from_quat(
            [
                t_base_link_2_camera.transform.rotation.x,
                t_base_link_2_camera.transform.rotation.y,
                t_base_link_2_camera.transform.rotation.z,
                t_base_link_2_camera.transform.rotation.w,
            ]
        ).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot_matrix
        transformation_matrix[:3, 3] = [
            t_base_link_2_camera.transform.translation.x,
            t_base_link_2_camera.transform.translation.y,
            t_base_link_2_camera.transform.translation.z,
        ]

        try:
            path = os.path.join(self.save_dir, "camera_poses", f"image_{self.image_count:04d}.txt")
            np.savetxt(path, transformation_matrix.tolist())
            self.get_logger().info("Saved camera pose")
        except Exception as e:
            self.get_logger().error(f"Failed to save camera pose: {e}")

    def point_cloud_callback(self, point_cloud):
        try:
            # Directory to save PCD files
            path = os.path.join(self.save_dir, "point_cloud", f"point_cloud_{self.image_count:04d}.pcd")
            
            # Lists to hold xyz and rgb data
            xyz = []
            rgb = []
            
            # Use sensor_msgs point_cloud2 utility to read the points from the PointCloud2 message
            pc_data = point_cloud2.read_points(point_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            
            # Process each point in the point cloud data
            for p in pc_data:
                x, y, z, rgb_value = p
                # Decode RGB value
                packed_rgb = struct.unpack('I', struct.pack('f', rgb_value))[0]  # Convert RGB from float32 to int32
                r = (packed_rgb & 0x00FF0000) >> 16
                g = (packed_rgb & 0x0000FF00) >> 8
                b = (packed_rgb & 0x000000FF)
                
                # Append point (x, y, z) and corresponding RGB values
                xyz.append([x, y, z])
                rgb.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB to 0-1 range for Open3D

            # Convert lists to numpy arrays for Open3D compatibility
            xyz = np.array(xyz)
            rgb = np.array(rgb)
            
            # Create Open3D point cloud object and set points and colors
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(xyz)
            out_pcd.colors = o3d.utility.Vector3dVector(rgb)
            
            # Write the point cloud to a PCD file
            o3d.io.write_point_cloud(path, out_pcd)
            
            self.get_logger().info("Received and saved point cloud data")

        except Exception as e:
            self.get_logger().error(f"Failed to process point cloud: {e}")


    # def publish_new_frame(self, name, pose: PoseStamped):
    #     t = TransformStamped()

    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = pose.header.frame_id
    #     t.child_frame_id = name
    #     t.transform.translation.x = pose.pose.position.x
    #     t.transform.translation.y = pose.pose.position.y
    #     t.transform.translation.z = pose.pose.position.z
    #     t.transform.rotation.x = pose.pose.orientation.x
    #     t.transform.rotation.y = pose.pose.orientation.y
    #     t.transform.rotation.z = pose.pose.orientation.z
    #     t.transform.rotation.w = pose.pose.orientation.w

    #     self.tf_static_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init()
    ic = DataCollector()
    try:
        rclpy.spin(ic)
    except KeyboardInterrupt:
        rclpy.spin_once(ic)
        rclpy.shutdown()
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
