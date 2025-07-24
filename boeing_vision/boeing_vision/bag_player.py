#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import time

class BagPlayer(Node):
    def __init__(self, bag_directory):
        super().__init__('bag_player')

        """
        self.declare_parameter('surface_type', 'flat')  # Default value is 'flat'
        valid_surface_types = ['flat', 'downhill', 'uphill']

        # Validate the parameter value
        #surface_type = self.get_parameter('surface_type').value
        #if surface_type not in valid_surface_types:
        #    raise ValueError(f"Invalid surface_type: {surface_type}. Must be one of {valid_surface_types}")
        self.declare_parameter('rock_id', 0)
        self.declare_parameter('run_id', 1)

        surface_type = self.get_parameter('surface_type').value
        rock_id = self.get_parameter('rock_id').value
        run_id = self.get_parameter('run_id').value

        # Add a callback for parameter updates
        self.add_on_set_parameters_callback(self.parameter_callback)
        """

        self.bag_publisher = self.create_publisher(String, 'current_bag_name', 10)
        self.bag_directory = bag_directory
        self.get_logger().info(f"Bag player initialized with directory: {bag_directory}")


    def play_bags(self):
        # Find all bag directories (ROS 2 bags are directories)
        bag_dirs = []
        for root, dirs, files in os.walk(self.bag_directory):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for bag_folder in os.listdir(dir_path):
                    bag_dir = os.path.join(dir_path, bag_folder)
                    if os.path.isdir(bag_dir):
                        bag_dirs.append(bag_dir)

        #print(bag_dirs)

        for bag_dir in bag_dirs:            
            # Publish the current bag name
            msg = String()
            bag_name = bag_dir.split('/')[-2]
            msg.data = bag_name
            self.bag_publisher.publish(msg)

            self.get_logger().info(f"Now playing: {bag_name}")
            
            # Play the bag
            process = subprocess.Popen(['ros2', 'bag', 'play', bag_dir], stdout=subprocess.DEVNULL,)
            
            # Wait for the bag to finish playing
            while process.poll() is None:
                time.sleep(0.1)
            
            self.get_logger().info(f"Finished playing: {bag_name}")
            
        self.get_logger().info("All bags played")

def main(args=None):
    rclpy.init(args=args)
    
    # Get bag directory from parameter
    node = BagPlayer('/home/students/girgine/Documents/boeing/real_world/in_lab_2')  # Update this path
    #node = BagPlayer('/home/students/girgine/Documents/boeing/perlin_rock810')  # Update this path

    
    # Play all bags
    node.play_bags()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()