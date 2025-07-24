import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
import numpy as np

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# Create a QoS profile with reliable reliability
qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    depth=10
)

class OrientationNode(Node):
    def __init__(self):
        super().__init__('orientation_node')
        
        # Initialize quaternion (no rotation)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Subscribe to IMU data (gyroscope)
        self.subscription = self.create_subscription(
            Imu,
            '/camera/camera/imu',  # Update this topic name if necessary
            self.imu_callback, qos_profile)
        
        # Publisher for orientation
        self.orientation_pub = self.create_publisher(Quaternion, '/orientation', qos_profile)
        
        # Time variables for integration
        self.last_time = self.get_clock().now()

        print('Orientation node initialized')

    def imu_callback(self, msg):
        # Get gyroscope data (angular velocity in rad/s)
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        # Calculate time step
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9  # Convert to seconds
        self.last_time = current_time
        
        # Update orientation using gyroscope data
        self.q = self.update_orientation(self.q, gyro, dt)
        
        # Publish orientation as a quaternion
        orientation_msg = Quaternion()
        orientation_msg.w = float(self.q[0])
        orientation_msg.x = float(self.q[1])
        orientation_msg.y = float(self.q[2])
        orientation_msg.z = float(self.q[3])
        self.orientation_pub.publish(orientation_msg)

    def update_orientation(self, q, gyro, dt):
        wx, wy, wz = gyro
        omega = np.array([[0, -wx, -wy, -wz],
                          [wx, 0, wz, -wy],
                          [wy, -wz, 0, wx],
                          [wz, wy, -wx, 0]])
        q_new = q + 0.5 * dt * np.dot(omega, q)
        return q_new / np.linalg.norm(q_new)  # Normalize quaternion

def main(args=None):
    rclpy.init(args=args)
    node = OrientationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()