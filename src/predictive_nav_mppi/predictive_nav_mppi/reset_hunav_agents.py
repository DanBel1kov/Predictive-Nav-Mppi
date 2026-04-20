#!/usr/bin/env python3
"""Teleport HuNav agents back to their initial poses from /get_agents."""

import math
from pathlib import Path
import subprocess

import rclpy
from rclpy.node import Node

import yaml

from geometry_msgs.msg import Pose
from hunav_msgs.msg import Agent
from hunav_msgs.msg import AgentBehavior
from hunav_msgs.msg import Agents
from hunav_msgs.srv import ResetAgents

try:
    from gazebo_msgs.msg import EntityState
    from gazebo_msgs.srv import GetEntityState
    from gazebo_msgs.srv import SetEntityState
except ImportError as exc:  # pragma: no cover - depends on ROS env
    raise SystemExit(f"gazebo_msgs is required: {exc}")


def _rpy_quat(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _actor_z(agent, actor_z_offset: float) -> float:
    # Mirror HuNavPlugin::UpdateGazeboPedestrians so reset pose matches runtime pose.
    skin_heights = {
        0: 0.96,
        1: 0.97,
        2: 0.93,
        3: 0.93,
        4: 0.97,
        5: 1.05,
        6: 1.05,
        7: 1.05,
        8: 1.05,
    }
    skin = int(getattr(agent, "skin", -1))
    if skin in skin_heights:
        return skin_heights[skin] + actor_z_offset
    return float(agent.position.position.z) + actor_z_offset


def _behavior_type(value: str) -> int:
    mapping = {
        "regular": AgentBehavior.BEH_REGULAR,
        "impassive": AgentBehavior.BEH_IMPASSIVE,
        "surprised": AgentBehavior.BEH_SURPRISED,
        "scared": AgentBehavior.BEH_SCARED,
        "curious": AgentBehavior.BEH_CURIOUS,
        "threatening": AgentBehavior.BEH_THREATENING,
    }
    return mapping.get(str(value).strip().lower(), AgentBehavior.BEH_REGULAR)


def _pose_goal(x: float, y: float) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0
    pose.orientation.w = 1.0
    return pose


class ResetHunavAgents(Node):
    def __init__(self):
        super().__init__("reset_hunav_agents")

        self.use_sim_time = bool(self._declare_or_get("use_sim_time", True))
        self.hunav_params_file = str(self._declare_or_get("hunav_params_file", ""))
        self.set_entity_state_service = str(
            self._declare_or_get("set_entity_state_service", "/gazebo/set_entity_state")
        )
        self.set_entity_state_service_alt = str(
            self._declare_or_get("set_entity_state_service_alt", "/set_entity_state")
        )
        self.get_entity_state_service = str(
            self._declare_or_get("get_entity_state_service", "/gazebo/get_entity_state")
        )
        self.get_entity_state_service_alt = str(
            self._declare_or_get("get_entity_state_service_alt", "/get_entity_state")
        )
        self.reset_agents_service = str(
            self._declare_or_get("reset_agents_service", "/reset_agents")
        )
        self.actor_z_offset = float(self._declare_or_get("actor_z_offset", -0.55))
        self.robot_model_name = str(self._declare_or_get("robot_model_name", "waffle"))
        self.robot_radius = float(self._declare_or_get("robot_radius", 0.35))

        if self.use_sim_time:
            self.set_parameters(
                [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)]
            )

        self._set_entity = self.create_client(SetEntityState, self.set_entity_state_service)
        self._set_entity_alt = self.create_client(
            SetEntityState, self.set_entity_state_service_alt
        )
        self._get_entity = self.create_client(GetEntityState, self.get_entity_state_service)
        self._get_entity_alt = self.create_client(
            GetEntityState, self.get_entity_state_service_alt
        )
        self._reset_agents = self.create_client(ResetAgents, self.reset_agents_service)

    def _declare_or_get(self, name: str, default):
        if self.has_parameter(name):
            return self.get_parameter(name).value
        return self.declare_parameter(name, default).value

    def _select_client(self, primary, alternate, label: str, timeout_sec: float = 3.0):
        if primary.wait_for_service(timeout_sec=timeout_sec):
            return primary
        if alternate is not None and alternate.wait_for_service(timeout_sec=timeout_sec):
            return alternate
        self.get_logger().warn(f"{label} service is not available")
        return None

    def _gz_model_teleport(self, agent) -> bool:
        # `gz model` sets the base actor pose, so it must match the raw pose
        # emitted by WorldGenerator (`x y z 0 0 yaw`). HuNavPlugin will then
        # apply its own runtime transform on the next update.
        yaw = float(agent.yaw)
        z = float(agent.position.position.z)
        cmd = [
            "gz", "model",
            "-m", str(agent.name),
            "-x", str(float(agent.position.position.x)),
            "-y", str(float(agent.position.position.y)),
            "-z", str(z),
            "-R", "0",
            "-P", "0",
            "-Y", str(yaw),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5.0, check=False
            )
        except FileNotFoundError:
            self.get_logger().error("'gz' CLI not found")
            return False
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"gz model timeout for {agent.name}")
            return False
        if result.returncode != 0:
            self.get_logger().warn(
                f"gz model failed for {agent.name}: {result.stderr.strip()}"
            )
            return False
        return True

    def _fetch_robot(self, get_entity_cli):
        if get_entity_cli is None:
            robot = Agent()
            robot.id = -1
            robot.type = Agent.ROBOT
            robot.name = self.robot_model_name
            robot.group_id = -1
            robot.position.orientation.w = 1.0
            robot.radius = float(self.robot_radius)
            robot.desired_velocity = 0.0
            robot.linear_vel = 0.0
            robot.angular_vel = 0.0
            robot.yaw = 0.0
            return robot
        req = GetEntityState.Request()
        req.name = self.robot_model_name
        req.reference_frame = "world"
        future = get_entity_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None or not future.result().success:
            self.get_logger().error(f"Failed to fetch robot state for {self.robot_model_name}")
            return None

        res = future.result()
        robot = Agent()
        robot.id = -1
        robot.type = Agent.ROBOT
        robot.name = self.robot_model_name
        robot.group_id = -1
        robot.position = res.state.pose
        robot.velocity = res.state.twist
        robot.radius = float(self.robot_radius)
        robot.desired_velocity = 0.0
        robot.linear_vel = 0.0
        robot.angular_vel = 0.0

        q = res.state.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        robot.yaw = float(math.atan2(siny_cosp, cosy_cosp))
        return robot

    def _load_initial_agents(self):
        if not self.hunav_params_file:
            self.get_logger().error("hunav_params_file parameter is required")
            return None
        cfg_path = Path(self.hunav_params_file)
        if not cfg_path.is_file():
            self.get_logger().error(f"HuNav params file not found: {cfg_path}")
            return None

        data = yaml.safe_load(cfg_path.read_text())
        params = data.get("hunav_loader", {}).get("ros__parameters", {})
        goal_map = params.get("global_goals", {})
        agent_names = params.get("agents", [])

        agents = []
        for name in agent_names:
            a = params.get(name, {})
            if not a:
                continue
            msg = Agent()
            msg.id = int(a.get("id", -1))
            msg.type = Agent.PERSON
            msg.skin = int(a.get("skin", -1))
            msg.name = str(name)
            msg.group_id = int(a.get("group_id", -1))

            init_pose = a.get("init_pose", {})
            msg.position.position.x = float(init_pose.get("x", 0.0))
            msg.position.position.y = float(init_pose.get("y", 0.0))
            msg.position.position.z = float(init_pose.get("z", 0.0))
            msg.position.orientation.w = 1.0
            msg.yaw = float(init_pose.get("h", 0.0))

            msg.velocity.linear.x = 0.0
            msg.velocity.linear.y = 0.0
            msg.velocity.angular.z = 0.0
            msg.desired_velocity = float(a.get("max_vel", 0.0))
            msg.radius = float(a.get("radius", 0.35))
            msg.linear_vel = 0.0
            msg.angular_vel = 0.0

            beh = a.get("behavior", {})
            msg.behavior.type = _behavior_type(beh.get("type", "Regular"))
            msg.behavior.state = AgentBehavior.BEH_ACTIVE_1
            msg.behavior.configuration = int(beh.get("configuration", 0))
            msg.behavior.duration = float(beh.get("duration", 40.0))
            msg.behavior.once = bool(beh.get("once", True))
            msg.behavior.vel = float(beh.get("vel", 1.0))
            msg.behavior.dist = float(beh.get("dist", 0.0))
            msg.behavior.social_force_factor = float(beh.get("social_force_factor", 5.0))
            msg.behavior.goal_force_factor = float(beh.get("goal_force_factor", 2.0))
            msg.behavior.obstacle_force_factor = float(beh.get("obstacle_force_factor", 10.0))
            msg.behavior.other_force_factor = float(beh.get("other_force_factor", 20.0))

            msg.cyclic_goals = bool(a.get("cyclic_goals", True))
            msg.goal_radius = float(a.get("goal_radius", 0.0))
            for goal_id in a.get("goals", []):
                goal = goal_map.get(int(goal_id), goal_map.get(str(goal_id), {}))
                if not goal:
                    continue
                msg.goals.append(_pose_goal(goal.get("x", 0.0), goal.get("y", 0.0)))
            agents.append(msg)
        return agents

    def run(self) -> int:
        agents = self._load_initial_agents()
        if agents is None:
            return 1

        set_entity_cli = self._select_client(
            self._set_entity, self._set_entity_alt, "SetEntityState"
        )
        get_entity_cli = self._select_client(
            self._get_entity, self._get_entity_alt, "GetEntityState"
        )
        if not self._reset_agents.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"Service not available: {self.reset_agents_service}")
            return 1

        if not agents:
            self.get_logger().warn("No agents parsed from hunav_params_file")
            return 0

        reset_ok = 0
        for agent in agents:
            state = EntityState()
            state.name = agent.name
            state.pose.position.x = float(agent.position.position.x)
            state.pose.position.y = float(agent.position.position.y)
            state.pose.position.z = _actor_z(agent, self.actor_z_offset)
            qx, qy, qz, qw = _rpy_quat(1.5707, 0.0, float(agent.yaw) + math.pi / 2.0)
            state.pose.orientation.x = qx
            state.pose.orientation.y = qy
            state.pose.orientation.z = qz
            state.pose.orientation.w = qw
            state.twist.linear.x = 0.0
            state.twist.linear.y = 0.0
            state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0
            state.twist.angular.y = 0.0
            state.twist.angular.z = 0.0
            state.reference_frame = "world"

            if set_entity_cli is not None:
                set_req = SetEntityState.Request()
                set_req.state = state
                set_future = set_entity_cli.call_async(set_req)
                rclpy.spin_until_future_complete(self, set_future, timeout_sec=5.0)
                if set_future.done() and set_future.result() is not None and set_future.result().success:
                    reset_ok += 1
                else:
                    self.get_logger().warn(f"Failed to reset agent {agent.name} via SetEntityState")
            elif self._gz_model_teleport(agent):
                reset_ok += 1
            else:
                self.get_logger().warn(f"Failed to reset agent {agent.name}")

        if reset_ok != len(agents):
            self.get_logger().error(f"Reset {reset_ok}/{len(agents)} HuNav agents")
            return 1

        robot = self._fetch_robot(get_entity_cli)
        if robot is None:
            return 1

        reset_req = ResetAgents.Request()
        reset_req.robot = robot
        reset_req.current_agents = Agents()
        reset_req.current_agents.agents = agents
        reset_req.current_agents.header.frame_id = "map"
        reset_req.current_agents.header.stamp = self.get_clock().now().to_msg()
        reset_future = self._reset_agents.call_async(reset_req)
        rclpy.spin_until_future_complete(self, reset_future, timeout_sec=10.0)
        if not reset_future.done() or reset_future.result() is None or not reset_future.result().ok:
            self.get_logger().error("reset_agents service call failed")
            return 1

        self.get_logger().info(f"Reset {reset_ok}/{len(agents)} HuNav agents and BT state")
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = ResetHunavAgents()
    try:
        code = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)
