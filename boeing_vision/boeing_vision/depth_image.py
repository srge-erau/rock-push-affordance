import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ament_index_python.packages import get_package_share_directory

class DepthImageSaver(Node):
    def __init__(self):
        super().__init__('depth_image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.image = None
        self.get_logger().info("Depth Image Saver node started. Press 's' to save the image.")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def save_image(self):
        if self.image is not None:
            # Create a directory to save images if it doesn't exist
            package_path = '/home/students/girgine/ros2_ws/src/boeing_vision'
            save_path = os.path.join(package_path, 'depth_images')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the image with a timestamp
            timestamp = self.get_clock().now().to_msg()
            filename = f"{save_path}/depth_image_{timestamp.sec}_{timestamp.nanosec}.png"
            cv2.imwrite(filename, self.image)
            self.get_logger().info(f"Image saved as {filename}")
        else:
            self.get_logger().warn("No image received yet.")

def main(args=None):
    rclpy.init(args=args)
    depth_image_saver = DepthImageSaver()

    try:
        while rclpy.ok():
            rclpy.spin_once(depth_image_saver, timeout_sec=0.1)
            # Check for user input
            key = input("Press 's' to save the image or 'q' to quit: ")
            if key == 's':
                depth_image_saver.save_image()
            elif key == 'q':
                break
    except KeyboardInterrupt:
        pass
    finally:
        depth_image_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()