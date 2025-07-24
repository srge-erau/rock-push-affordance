import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from boeing_interfaces.msg import SegmentedPointCloud
from std_msgs.msg import Header
import numpy as np
import struct
import random
import open3d as o3d
from sklearn.cluster import KMeans  # Import KMeans from scikit-learn
from example_interfaces.msg import Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3

class PointCloudColorizer(Node):

    def __init__(self):
        super().__init__('pointcloud_colorizer')

        # Subscriber to the input PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            #'/velodyne_points',  # Change this to your input topic
            '/filtered_pointcloud',
            self.pointcloud_callback,
            10)

        # Publisher for the colored PointCloud2
        self.pc_publisher = self.create_publisher(
            PointCloud2,
            '/colored_pointcloud',  # Output topic
            10)
        
        # Publisher for the colored PointCloud2
        self.segmented_publisher = self.create_publisher(
            SegmentedPointCloud,
            '/segmented_pointcloud',  # Output topic
            10)
        
        # Publisher for bounding box markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/normals',
            10)

        
        # 0.97 threshold 0.965 - 0.96
        # 0.5 radius 0.3 - 0.2
        # 10 max_nn 5 - 5

        # old values 0.93, 0.2, 5 for cosine_threshold, normal_search_radius, max_nn

        # Read parameters from ROS parameters
        self.cosine_threshold = self.declare_parameter('cosine_threshold', 0.88).value  # Higher threshold means less ground
        self.normal_search_radius = self.declare_parameter('normal_search_radius', 0.5).value  # Search radius for normal estimation
        self.max_nn = self.declare_parameter('max_nn', 30).value  # Maximum number of nearest neighbors for normal estimation. Higher values make ground to be classified as object
        
        # old values 0.15, 10 for dbscan_eps, dbscan_min_points

        # DBSCAN parameters
        self.dbscan_eps = self.declare_parameter('dbscan_eps', 0.15).value  # DBSCAN epsilon
        self.dbscan_min_points = self.declare_parameter('dbscan_min_points', 15).value  # DBSCAN minimum points

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Generate a color map for 10 elements
        self.color_map = [
            #(255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (0, 0, 255),  # Blue
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple

        ] #[tuple(random.randint(0, 255) for _ in range(3)) for _ in range(10)]

        self.get_logger().info('PointCloud Colorizer Node has started.')

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'cosine_threshold' and param.type_ == param.Type.DOUBLE:
                self.cosine_threshold = param.value
            elif param.name == 'normal_search_radius' and param.type_ == param.Type.DOUBLE:
                self.normal_search_radius = param.value
            elif param.name == 'max_nn' and param.type_ == param.Type.INTEGER:
                self.max_nn = param.value
            elif param.name == 'dbscan_eps' and param.type_ == param.Type.DOUBLE:
                self.dbscan_eps = param.value
            elif param.name == 'dbscan_min_points' and param.type_ == param.Type.INTEGER:
                self.dbscan_min_points = param.value
        return rclpy.node.SetParametersResult(successful=True)

    def pointcloud_callback(self, msg):

        ########################## READ POINT CLOUD DATA ##########################
        # Convert PointCloud2 to a numpy array
        points = np.frombuffer(msg.data, dtype=np.uint8)
        points = points.reshape(-1, msg.point_step)

        # Extract XYZ coordinates (assuming the point format is XYZ)
        xyz = np.array([struct.unpack('fff', point[:12]) for point in points])

        #xyz = xyz[xyz[:, 0] >= 0]

        # Convert to Open3D point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz)


        ########################## NORMAL ESTIMATION ##########################
        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_search_radius, max_nn=self.max_nn))
        
        # Visualize the estimated normals
        #self.pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 10]))
        #o3d.visualization.draw_geometries([self.pcd], point_show_normal=True)

        # Get the normals as a numpy array
        normals = np.asarray(self.pcd.normals)
        normals[normals[:, 2] < 0] *= -1 # Flip normals to point upwards

        ########################## NORMAL VISUALIZATION ##########################
        
        marker_array = MarkerArray()

        assert len(normals) == len(xyz), "Number of normals must be equal to the number of points"

        for n_id in range(len(normals)):
            normal_marker = self.create_normal_marker(
                np.asarray(self.pcd.points)[n_id], 
                np.asarray(normals)[n_id], 
                n_id, 
                color=(0.0, 1.0, 0.0), 
                frame=msg.header.frame_id
            )
            marker_array.markers.append(normal_marker)
        self.marker_pub.publish(marker_array)
        

        ########################## SEGMENTATION ##########################
        
        # Separate ground and object points
        ground_mask, object_mask = self.cluster_cosine_similarity(normals)

        # segment the object points into individual objects
        indexed_points = np.column_stack((np.arange(len(xyz), dtype=int), xyz))
        object_points = indexed_points[object_mask]
        object_clusters = self.cluster_objects(object_points)

        # Assign colors based on the clusters
        colors = np.zeros(len(xyz))
        colors[ground_mask] = 0  # Ground (red)
        for cluster_id, cluster in enumerate(object_clusters):
            colors[cluster] = cluster_id + 1

        ########################## PUBLISH SEGMENTED POINT CLOUD ##########################

        # Create a new PointCloud2 message for the colored points
        colored_points = PointCloud2()
        colored_points.header = msg.header
        colored_points.height = msg.height
        colored_points.width = msg.width
        colored_points.fields = msg.fields
        colored_points.is_bigendian = msg.is_bigendian
        colored_points.point_step = msg.point_step + 4  # Adding 4 bytes for the color
        colored_points.row_step = colored_points.point_step * colored_points.width
        colored_points.is_dense = msg.is_dense


        ############################# ADD COLOR TO POINTS #############################
        # Add a new field for the color (RGBA)
        colored_points.fields.append(PointField(name='rgba', offset=msg.point_step, datatype=PointField.UINT32, count=1))

        # Create a new buffer for the colored points
        colored_data = bytearray()

        for i, point in enumerate(points):
            # Append the original point data
            colored_data.extend(point)
            # Get the color for the point based on its segment label
            rgba = self.get_segment_color(colors[i])
            colored_data.extend(rgba)

        colored_points.data = bytes(colored_data)

        ############################# PUBLISH POINT CLOUD #############################
        # Publish the colored point cloud
        self.pc_publisher.publish(colored_points)

        segmented_points = SegmentedPointCloud()
        segmented_points.pointcloud = colored_points
        segmented_points.segment_ids = colors.astype(int).tolist()
        self.segmented_publisher.publish(segmented_points)
        


    def cluster_cosine_similarity(self, normals):
        """
        Clusters normals into ground and object based on cosine similarity.
        - Ground normals are parallel or antiparallel to the reference normal.
        - Object normals are non-parallel.
        """
        # Compute the reference normal (e.g., the average normal of the ground)
        reference_normal = np.median(normals, axis=0)
        reference_normal /= np.linalg.norm(reference_normal)  # Normalize

        # Compute cosine similarity between each normal and the reference normal
        cosine_similarity = np.abs(np.dot(normals, reference_normal))

        # Cluster normals based on the cosine similarity threshold
        ground_mask = cosine_similarity > self.cosine_threshold
        object_mask = ~ground_mask

        return ground_mask, object_mask
    

    def cluster_objects(self, object_points):
        """
        Clusters object points into individual objects using Euclidean clustering.
        - Uses Open3D's DBSCAN implementation.
        - Returns a list of clusters, where each cluster is an array of indices.
        """
        # Convert object points to Open3D point cloud
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points[:, 1:]) # select only the points but not indices

        # Perform Euclidean clustering
        labels = np.array(object_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))

        # Group points into clusters
        clusters = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_indices = object_points[:,0][np.where(labels == label)[0]]
            clusters.append(cluster_indices.astype(int))

        return clusters
    

    def get_segment_color(self, label):
        """
        Assigns a color to a point based on its segment label.
        - Ground (label = 0): Blue
        - Object (label > 0): Random color from the color map
        """
        if label == 0:
            # Ground (red)
            r, g, b, a = 0, 0, 255, 255
        else:
            # Object (random color)
            b, g, r = self.color_map[int(label % len(self.color_map))]
            a = 255  # encode the label into the alpha channel

        # Pack the color into a 4-byte RGBA format
        rgba = struct.pack('BBBB', r, g, b, a)
        return rgba
    
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
        end_point = Point(x=centroid[0] + normal[0], y=centroid[1] + normal[1], z=centroid[2]+normal[2])

        normal_marker.points.append(start_point)
        normal_marker.points.append(end_point)

        return normal_marker

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudColorizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
