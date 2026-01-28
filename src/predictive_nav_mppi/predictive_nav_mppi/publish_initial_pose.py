import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')

        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.sent_count = 0
        self.max_sends = 5
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        if self.get_clock().now().nanoseconds == 0:
            self.get_logger().info('Waiting for /clock (sim time is 0)...')
            return

        if self.pub.get_subscription_count() == 0:
            self.get_logger().info('Waiting for AMCL subscription on /initialpose...')
            return

        x = float(self.get_parameter('x').value)
        y = float(self.get_parameter('y').value)
        yaw = float(self.get_parameter('yaw').value)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.0685
        msg.pose.covariance = cov

        self.pub.publish(msg)
        self.sent_count += 1
        self.get_logger().info(f'Published /initialpose #{self.sent_count}')

        if self.sent_count >= self.max_sends:
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    rclpy.spin(node)
    # на случай если spin завершится без shutdown
    if rclpy.ok():
        rclpy.shutdown()
