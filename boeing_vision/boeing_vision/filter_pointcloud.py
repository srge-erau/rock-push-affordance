#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct


class PointCloudFilterNode(Node):
    def __init__(self):
        super().__init__('pointcloud_filter_node')
        
        # Subscriber to the PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',  # Replace with your input topic
            self.pointcloud_callback,
            10)
        
        # Publisher for the filtered PointCloud2
        self.publisher = self.create_publisher(
            PointCloud2,
            '/filtered_pointcloud',  # Replace with your output topic
            10)
        
        # Thresholds for filtering
        self.z_threshold = 2.0  # Filter points with Z > 2.0
        self.distance_threshold = 5.0  # Filter points farther than 5.0 units from origin

        # Downsampling voxel size
        self.voxel_size = 0.05  # Adjust based on your requirements
        
        self.get_logger().info("PointCloud Filter Node has started.")

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to a numpy array
        points = np.frombuffer(msg.data, dtype=np.uint8)
        points = points.reshape(-1, msg.point_step)

        # Extract XYZ coordinates (assuming the point format is XYZ)
        xyz = np.array([struct.unpack('fff', point[:12]) for point in points])

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Downsample the point cloud using voxel downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Rotate the point cloud by -90 degrees around the X-axis
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]])
        rotated_points = np.dot(downsampled_points, rotation_matrix.T)

        # Apply filtering to discard points with Y > 2
        y_filter = rotated_points[:, 1] <= 2
        filtered_points = rotated_points[y_filter]
        
        """
        selection_polygon = o3d.visualization.read_selection_polygon_volume("/home/students/girgine/ros2_ws/src/boeing_vision/config/pc_crop.json")
        cropped_pcd = selection_polygon.crop_point_cloud(pcd)

        o3d.visualization.draw_geometries([cropped_pcd])
        """

        # Convert filtered points back to ROS PointCloud2 message
        filtered_msg = point_cloud2.create_cloud_xyz32(msg.header, filtered_points)
        
        # Publish the filtered point cloud
        self.publisher.publish(filtered_msg)
        self.get_logger().info("Published filtered point cloud.")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()