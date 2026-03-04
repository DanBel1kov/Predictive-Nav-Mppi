#!/usr/bin/env python3

import copy
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from hunav_msgs.srv import ComputeAgents


class ComputeAgentsProxy(Node):
    def __init__(self) -> None:
        super().__init__('compute_agents_proxy')

        self.declare_parameter('backend_service', 'compute_agents_raw')
        self.declare_parameter('frontend_service', 'compute_agents')
        self.declare_parameter('robot_mask_distance', 10000.0)

        backend_service = self.get_parameter('backend_service').value
        frontend_service = self.get_parameter('frontend_service').value
        self._robot_mask_distance = float(
            self.get_parameter('robot_mask_distance').value)

        self._cb_group = ReentrantCallbackGroup()

        self._backend_client = self.create_client(
            ComputeAgents, backend_service, callback_group=self._cb_group)
        self._service = self.create_service(
            ComputeAgents, frontend_service, self._handle_compute_agents,
            callback_group=self._cb_group)

        self.get_logger().info(
            f'ComputeAgents proxy started: {frontend_service} -> {backend_service}')

    def _mask_robot(self, request: ComputeAgents.Request) -> ComputeAgents.Request:
        proxied = ComputeAgents.Request()
        proxied.current_agents = request.current_agents
        proxied.robot = copy.deepcopy(request.robot)

        # Push the robot far away so humans ignore it in social force.
        proxied.robot.position.position.x = self._robot_mask_distance
        proxied.robot.position.position.y = self._robot_mask_distance
        proxied.robot.velocity.linear.x = 0.0
        proxied.robot.velocity.linear.y = 0.0
        proxied.robot.velocity.angular.z = 0.0
        proxied.robot.linear_vel = 0.0
        proxied.robot.angular_vel = 0.0
        return proxied

    def _handle_compute_agents(
            self,
            request: ComputeAgents.Request,
            response: ComputeAgents.Response) -> ComputeAgents.Response:
        if not self._backend_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Backend service compute_agents_raw is unavailable')
            response.updated_agents = request.current_agents
            return response

        proxied_request = self._mask_robot(request)
        try:
            # With MultiThreadedExecutor + Reentrant callback group this call is stable
            # and preserves HuNav's normal human-human interaction dynamics.
            result = self._backend_client.call(proxied_request)
            response.updated_agents = result.updated_agents
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'Backend compute_agents_raw failed: {exc}')
            response.updated_agents = request.current_agents
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ComputeAgentsProxy()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

