import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from boeing_interfaces.msg import SegmentedPointCloud, Obstacle, ObstacleList
import numpy as np
import struct
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import String


import numpy as np
from scipy.optimize import least_squares
import csv, math

class ObstacleFeatureExtractor(Node):

    def __init__(self):
        super().__init__('obstacle_feature_extractor')

        # Subscriber to the colored point cloud topic
        self.subscription = self.create_subscription(
            SegmentedPointCloud,
            '/segmented_pointcloud',  # Topic from the previous node
            self.pointcloud_callback,
            10)
        
        # Publisher for bounding box markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/bounding_boxes',  # Topic for visualizing bounding boxes
            10)
        
        self.obstacle_pub = self.create_publisher(
            ObstacleList,
            '/obstacle_list',  # Topic for publishing the list of obstacles
            10)
        
        self.subscription = self.create_subscription(
            String,
            'current_bag_name',
            self.bag_name_callback,
            10)

        # Read parameters from ROS parameters
        self.declare_parameter('bbox_scale', 0.3)
        self.declare_parameter('normal_search_radius', 0.5)
        self.declare_parameter('max_nn', 30)
        self.declare_parameter('theta_tolerance', 1.0)

        self.bbox_scale = self.get_parameter('bbox_scale').value
        self.normal_search_radius = self.get_parameter('normal_search_radius').value
        self.max_nn = self.get_parameter('max_nn').value
        self.theta_tolerance = self.get_parameter('theta_tolerance').value  # Retrieve the new parameter

        # Add a callback for parameter updates
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.get_logger().info('Obstacle Feature Extractor Node has started.')

        self.records = []
        self.current_bag = 'unknown'
        self.sequence_normals = []
        self.majority_label = 'unknown'

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'bbox_scale' and param.type_ == param.Type.DOUBLE:
                self.bbox_scale = param.value
            elif param.name == 'normal_search_radius' and param.type_ == param.Type.DOUBLE:
                self.normal_search_radius = param.value
            elif param.name == 'max_nn' and param.type_ == param.Type.INTEGER:
                self.max_nn = param.value
            elif param.name == 'theta_tolerance' and param.type_ == param.Type.DOUBLE:  # Handle the new parameter
                self.theta_tolerance = param.value
        self.get_logger().info('Parameters updated.')
        return rclpy.node.SetParametersResult(successful=True)

    def pointcloud_callback(self, msg):
        """
        Callback function for processing incoming point cloud messages.
        This function processes a PointCloud2 message to extract obstacle features, 
        including bounding box dimensions, volume, shape, and surface normals. It 
        also calculates likelihood scores for obstacles based on their size and 
        orientation and publishes the results as markers and obstacle lists.
        Args:
            msg (PointCloud2): The incoming point cloud message containing segment IDs 
                               and point cloud data.
        Process:
            1. Extracts segment IDs and point cloud data from the message.
            2. Converts the PointCloud2 data into a numpy array for processing.
            3. Identifies ground points and clusters based on segment IDs.
            4. Fits ellipsoids to clusters and calculates bounding box dimensions, 
               volume, and center.
            5. Computes surface normals for ground points within the bounding box footprint.
            6. Calculates angles between normals and a reference vector to classify 
               normals as uphill, downhill, or ground.
            7. Assigns likelihood scores to obstacles based on their volume and orientation.
            8. Creates markers for bounding boxes and normals for visualization.
            9. Records obstacle features and normal data into a CSV file.
        Notes:
            - The function uses Open3D for point cloud processing and ROS for publishing 
              markers and obstacle lists.
            - The likelihood score thresholds (`T_high` and `T_low`) and other parameters 
              like `ground_margin`, `theta_tolerance`, and `normal_search_radius` are 
              configurable.
        Record Structure:
            Each record contains the following elements:
            - Obstacle position (x, y, z)
            - Bounding box dimensions (width, height, length)
            - Bounding box volume
            - Obstacle shape
            - Ground point coordinates (x, y, z) for each normal
            - Normal vector components (x, y, z) for each normal
            - Angle (theta) between the normal and the reference vector
        """

        segment_ids = np.asarray(msg.segment_ids)
        point_cloud = msg.pointcloud

        # Convert PointCloud2 to a numpy array
        points = np.frombuffer(point_cloud.data, dtype=np.uint8)
        points = points.reshape(-1, point_cloud.point_step)

        # Extract XYZ coordinates and RGBA colors
        xyz = np.array([struct.unpack('fff', point[:12]) for point in points])
    
        ground_indices = (segment_ids == 0)
        ground_points = xyz[ground_indices]

        marker_array = MarkerArray()
        obstacle_list = ObstacleList()

        # Loop through each rock cluster
        for cluster_id in np.unique(segment_ids):
            if cluster_id == 0:
                continue
            
            current_obstacle = Obstacle()
            current_obstacle.outcrop_flag = False

            # Get the points of the current cluster
            cluster_mask = (segment_ids == cluster_id)

            cluster_points = xyz[cluster_mask]

            _, shape = self.fit_ellipsoid_3d(cluster_points)
            current_obstacle.shape = shape

            # Convert cluster points to Open3D point cloud
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            # Compute the axis-aligned bounding box
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            
            current_obstacle.width = bbox.max_bound[0] - bbox.min_bound[0]
            current_obstacle.height = bbox.max_bound[1] - bbox.min_bound[1]
            current_obstacle.length = bbox.max_bound[2] - bbox.min_bound[2]

            # Calculate the volume of the bounding box
            bbox_volume = bbox.volume()
            current_obstacle.volume = bbox_volume

            # Calculate the center of the bounding box in 3D
            bbox_center = bbox.get_center()
            current_obstacle.position = Point(x=bbox_center[0], y=bbox_center[1], z=bbox_center[2])

            # Add a margin to the bounding box
            margin = self.bbox_scale  # Adjust the margin size as needed
            bbox_with_margin = bbox.scale(1 + margin, bbox_center)

            # Create a marker for the bounding box
            marker = self.create_bounding_box_marker(bbox_with_margin, int(cluster_id), frame=msg.pointcloud.header.frame_id)  # Red color for bounding box
            marker_array.markers.append(marker)

            bbox_corners = np.asarray(bbox_with_margin.get_box_points())
            footprint = bbox_corners[[0, 1, 7, 2]] # shape: (4, 3)

            ground_points_inside_footprint = self.get_points_inside_footprint(footprint, ground_points)

            # Calculate the normal of the ground points inside the footprint
            if len(ground_points_inside_footprint) >= 3:
                ground_pcd = o3d.geometry.PointCloud()
                ground_pcd.points = o3d.utility.Vector3dVector(ground_points_inside_footprint)
                ground_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_search_radius, max_nn=self.max_nn))

                # Rotate normals with negative z values by 180 degrees
                normals = np.asarray(ground_pcd.normals)
                normals[normals[:, 2] < 0] *= -1
                normals /= 3
                
                # Calculate the angle between bbox_center and ground_normal
                bbox_center_vector = np.array([bbox_center[0], bbox_center[1], bbox_center[2]])
                bbox_center_vector /= np.linalg.norm(bbox_center_vector)  # Normalize the vector

                #theta = np.arccos(np.clip(np.dot(normals, bbox_center_vector), -1.0, 1.0))
                theta = np.arccos(np.clip(np.dot(normals, [0.0, 1.0, 0.0]), -1.0, 1.0))
                theta_degrees = np.degrees(theta)

                color_array = []
                cluster_normals = []
                for t in theta_degrees:
                    if t >= (90 + self.theta_tolerance):  # uphill normals
                        color_array.append((0.0, 0.0, 1.0))
                        self.sequence_normals.append("uphill")
                        cluster_normals.append('uphill')
                    elif t <= (90 - self.theta_tolerance):  # downhill normals
                        color_array.append((0.0, 1.0, 0.0))
                        self.sequence_normals.append("downhill")
                        cluster_normals.append('downhill')
                    else:  # ground normals
                        color_array.append((1.0, 0.0, 0.0))
                        self.sequence_normals.append("ground")
                        cluster_normals.append('ground')

                # Find the majority category
                self.majority_label = max(set(self.sequence_normals), key=self.sequence_normals.count)
                self.get_logger().info(f"Sequence Label: {self.majority_label}")

                # Get all normals having the majority category
                #majority_normals = [normals[i] for i in range(len(normals)) if cluster_normals[i] == self.majority_label]

                # Get the thetas whose normal belongs to the majority label
                majority_thetas = [theta[i] for i in range(len(normals)) if cluster_normals[i] == self.majority_label]

                if len(majority_thetas) == 0 :
                    self.get_logger().info(f"No normals of {self.majority_label}. Skipping...")
                    continue

                # Get the points whose normal belongs to the majority label
                #majority_points = [np.asarray(ground_pcd.points)[i] for i in range(len(normals)) if normal_labels[i] == majority_label]
                
                mean_theta = np.mean(majority_thetas)

                for n_id in range(len(normals)):
                    normal_marker = self.create_normal_marker(np.asarray(ground_pcd.points)[n_id], 
                                                              np.asarray(normals)[n_id], n_id, color=color_array[n_id],  
                                                              frame=msg.pointcloud.header.frame_id)
                    marker_array.markers.append(normal_marker)

                record = [
                        current_obstacle.position.x,
                        current_obstacle.position.y,
                        current_obstacle.position.z,
                        current_obstacle.width, 
                        current_obstacle.height, 
                        current_obstacle.length, 
                        current_obstacle.volume * 10**4, 
                        current_obstacle.shape * 10**12,
                        mean_theta
                    ]
                #for n_id in range(len(majority_normals)):
                #    record.append(majority_points[n_id][0]),
                #    record.append(majority_points[n_id][1]),
                #    record.append(majority_points[n_id][2]),
                #    record.append(majority_normals[n_id][0])
                #    record.append(majority_normals[n_id][1])
                #    record.append(majority_normals[n_id][2])
                #    record.append(majority_thetas[n_id])

                self.records.append(record)

            obstacle_list.position.append(current_obstacle)
            
        self.obstacle_pub.publish(obstacle_list)
        self.marker_pub.publish(marker_array)
        self.save_to_csv(f'/home/students/girgine/ros2_ws/src/boeing_vision/dataset/{self.current_bag}.csv', self.records)


    def save_to_csv(self, filename, data):
        """
        Save a 2D list to a CSV file.

        Parameters:
        filename (str): The name of the CSV file to save.
        data (list of list): The 2D list to save into the CSV file.
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        #self.get_logger().info(f'Data saved to {filename}')

    def bag_name_callback(self, msg):
        self.current_bag = msg.data
        self.records = []
        self.get_logger().info(f"Received new bag name: {self.current_bag}")

        self.sequence_normals = []
        self.get_logger().info(f'Previous label was: {self.majority_label}')
        self.majority_label = 'unknown'


    def fit_ellipsoid_3d(self, points):
        """
        Fit a 3D ellipsoid to a set of points using least-squares optimization.

        Parameters:
        points (numpy.ndarray): A Nx3 array of points (x, y, z) to fit the ellipsoid to.

        Returns:
        tuple: A tuple containing the ellipsoid coefficients (A, B, C, D, E, F, G, H, I, J)
            and the root mean residual error.
        """
        def ellipsoid_residuals(coeffs, points):
            """
            Calculate the residuals for the ellipsoid fitting.
            """
            A, B, C, D, E, F, G, H, I, J = coeffs
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            return A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J

        # Initial guess for the ellipsoid coefficients (A, B, C, D, E, F, G, H, I, J)
        initial_guess = np.ones(10)

        # Perform least-squares optimization
        result = least_squares(ellipsoid_residuals, initial_guess, args=(points,))

        # Extract the optimized coefficients
        ellipsoid_coeffs = result.x

        # Calculate the root mean residual error
        residuals = ellipsoid_residuals(ellipsoid_coeffs, points)
        rmse = np.sqrt(np.mean(residuals**2))

        return ellipsoid_coeffs, rmse


    def get_points_inside_footprint(self, bbox_with_margin, ground_points):

        min_x, min_y = np.min(bbox_with_margin[:, :2], axis=0)
        max_x, max_y = np.max(bbox_with_margin[:, :2], axis=0)

        # Create masks for X and Y dimensions
        x_mask = (ground_points[:, 0] >= min_x) & (ground_points[:, 0] <= max_x)
        y_mask = (ground_points[:, 1] >= min_y) & (ground_points[:, 1] <= max_y)

        # Combine masks to find points inside the bounding box
        inside_mask = x_mask & y_mask
        
        # Return points inside the bounding box
        return ground_points[inside_mask]
    

    def create_normal_marker(self, centroid, normal, id, color=(0.0, 0.0, 1.0), frame='velodyne'):
         # Create a marker for the ground normal
        normal_marker = Marker()
        normal_marker.header.frame_id = frame  # Change this to your frame ID
        normal_marker.header.stamp = self.get_clock().now().to_msg()
        normal_marker.ns = "ground_normals"
        normal_marker.id = int(id)
        normal_marker.type = Marker.ARROW
        normal_marker.scale.x = 0.01  # Shaft diameter
        normal_marker.scale.y = 0.02  # Head diameter
        normal_marker.scale.z = 0.02  # Head length

        # Set the color of the normal arrow
        normal_marker.color.r = color[0]
        normal_marker.color.g = color[1]
        normal_marker.color.b = color[2]
        normal_marker.color.a = 1.0  # Fully opaque

        # Define the start and end points of the arrow
        start_point = Point(x=centroid[0], y=centroid[1], z=centroid[2])
        end_point = Point(x=centroid[0] + normal[0], y=centroid[1] + normal[1], z=centroid[2] + normal[2])

        normal_marker.points.append(start_point)
        normal_marker.points.append(end_point)

        return normal_marker


    def create_bounding_box_marker(self, bbox, id, frame='velodyne'):
        """
        Creates a Marker message for visualizing the bounding box.
        """
        marker = Marker()
        marker.header.frame_id = frame  # Change this to your frame ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = id
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.02  # Line thickness

        # Set the color of the bounding box
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        # Get the 8 corner points of the bounding box
        points = np.asarray(bbox.get_box_points())

        # Define the 12 edges of the bounding box using the 8 points
        edges = [
            (0, 1), (1, 7), (7, 2), (2, 0),  # Bottom face
            (3, 6), (6, 4), (4, 5), (5, 3),  # Top face
            (0, 3), (1, 6), (7, 4), (2, 5)   # Vertical edges
        ]

        # Add the points to the marker
        for edge in edges:
            p1 = points[edge[0]]
            p2 = points[edge[1]]
            marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
            marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))

        return marker
    

    


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleFeatureExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()