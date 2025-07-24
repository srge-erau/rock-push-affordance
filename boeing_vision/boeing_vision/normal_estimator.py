import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter

class SurfaceNormalsNode(Node):
    def __init__(self):
        super().__init__('surface_normals_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscribe to the input image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',  # Replace with your input image topic
            self.image_callback,
            1)
        
        # Publisher for the processed image (surface normals)
        self.publisher = self.create_publisher(
            Image,
            '/surface_normals',  # Replace with your output topic
            1)
        
    def finite_difference(self, depth_map):
        # Compute gradients using finite differences
        grad_x = np.gradient(depth_map, axis=1)  # Partial derivative wrt x
        grad_y = np.gradient(depth_map, axis=0)  # Partial derivative wrt y

        # Compute surface normals
        normals = np.dstack((-grad_x, -grad_y, np.ones_like(depth_map)))
        
        # Normalize the normals
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals_normalized = normals / norm

        return normals_normalized
    
    def gradient_based(self, depth_map):

        # Compute gradients using Sobel operators
        dx = sobel(depth_map, axis=1, mode='constant')  # Partial derivative w.r.t. x
        dy = sobel(depth_map, axis=0, mode='constant')  # Partial derivative w.r.t. y
        
        # Compute surface normals
        normals = np.dstack([-dx, -dy, np.ones_like(depth_map)])  # [ -D_x, -D_y, 1 ]
        
        # Normalize the normals to unit length
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / norm
        
        return normals
    
    def cross_product_based(self, depth_map):

        # Compute gradients using finite differences
        dx = np.gradient(depth_map, axis=1)  # Partial derivative w.r.t. x
        dy = np.gradient(depth_map, axis=0)  # Partial derivative w.r.t. y
        
        # Construct tangent vectors
        Tx = np.dstack([np.ones_like(depth_map), np.zeros_like(depth_map), dx])  # [1, 0, D_x]
        Ty = np.dstack([np.zeros_like(depth_map), np.ones_like(depth_map), dy])  # [0, 1, D_y]
        
        # Compute cross product to get normals
        normals = np.cross(Tx, Ty)
        
        # Normalize the normals to unit length
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / norm
        
        return normals
    
    def smooth_normals(self, normals, sigma=1.0):
        # Apply Gaussian smoothing to each component of the normals
        smoothed_normals = np.zeros_like(normals)
        for i in range(3):  # Iterate over x, y, z components
            smoothed_normals[:, :, i] = gaussian_filter(normals[:, :, i], sigma=sigma)
        
        # Normalize the smoothed normals to ensure they remain unit vectors
        norm = np.linalg.norm(smoothed_normals, axis=2, keepdims=True)
        smoothed_normals = smoothed_normals / norm
        
        return smoothed_normals

    def visualize_normals(self, normals):
        """ # Map normal vectors to RGB colors
        normals_visual = (normals + 1) / 2  # Shift from [-1, 1] to [0, 1]
        normals_visual = np.clip(normals_visual, 0, 1)  # Clip values to ensure they are within [0, 1]
        normals_visual = (normals_visual * 255).astype(np.uint8)  # Convert to 8-bit image """

        # Compute the angle (hue) from the x and y components of the normals
        angle = np.arctan2(normals[:, :, 1], normals[:, :, 0])  # Angle in radians
        hue = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        saturation = np.ones_like(hue)  # Full saturation
        value = np.ones_like(hue)  # Full brightness

        # Combine into an HSV image
        hsv_image = np.dstack([hue, saturation, value])

        # Convert HSV to RGB
        normals_visual = (cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8)

        return normals_visual

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Convert to grayscale if necessary
            if len(cv_image.shape) > 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Compute surface normals
            normals = self.finite_difference(cv_image)

            normals = self.smooth_normals(normals, sigma=5.0)
            
            # Visualize the normals
            normals_visual = self.visualize_normals(normals)
        
            # Convert the OpenCV image back to a ROS2 Image message
            normals_msg = self.bridge.cv2_to_imgmsg(normals_visual, encoding='rgb8')
            
            # Publish the processed image
            self.publisher.publish(normals_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    surface_normals_node = SurfaceNormalsNode()
    rclpy.spin(surface_normals_node)
    surface_normals_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()