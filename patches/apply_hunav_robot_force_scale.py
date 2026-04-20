#!/usr/bin/env python3
"""
Apply robot_force_scale patch to hunav_agent_manager.

This script adds:
  1. robot_force_scale_ parameter to AgentManager (agent_manager.hpp/.cpp)
  2. setRobotForceScale() wrapper to BTfunctions (bt_functions.hpp)
  3. robot_force_scale ROS parameter + /human_robot_forces publisher to BTnode (bt_node.hpp/.cpp)
  4. std_msgs dependency (CMakeLists.txt, package.xml)

Usage:
    python3 patches/apply_hunav_robot_force_scale.py /path/to/hunav_ws

After running, rebuild:
    cd /path/to/hunav_ws && colcon build --packages-select hunav_agent_manager
"""

import sys
import os

def patch_file(path, replacements):
    """Apply a list of (old, new) string replacements to a file."""
    with open(path, 'r') as f:
        content = f.read()

    applied = []
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            applied.append("OK")
        elif new in content:
            applied.append("ALREADY_APPLIED")
        else:
            applied.append(f"MISSING: {repr(old[:60])}")

    with open(path, 'w') as f:
        f.write(content)

    return applied


def main():
    if len(sys.argv) < 2:
        print("Usage: apply_hunav_robot_force_scale.py <hunav_ws_path>")
        sys.exit(1)

    ws = sys.argv[1]
    pkg = os.path.join(ws, "src", "hunav_sim", "hunav_agent_manager")

    files = {
        "agent_manager.hpp": os.path.join(pkg, "include", "hunav_agent_manager", "agent_manager.hpp"),
        "agent_manager.cpp": os.path.join(pkg, "src", "agent_manager.cpp"),
        "bt_functions.hpp":  os.path.join(pkg, "include", "hunav_agent_manager", "bt_functions.hpp"),
        "bt_node.hpp":       os.path.join(pkg, "include", "hunav_agent_manager", "bt_node.hpp"),
        "bt_node.cpp":       os.path.join(pkg, "src", "bt_node.cpp"),
        "CMakeLists.txt":    os.path.join(pkg, "CMakeLists.txt"),
        "package.xml":       os.path.join(pkg, "package.xml"),
    }

    patches = {
        # ── agent_manager.hpp ──────────────────────────────────────────────
        "agent_manager.hpp": [
            # Add setRobotForceScale() after getAgentForces()
            (
                "  sfm::Forces getAgentForces(int id)\n"
                "  {\n"
                "    return agents_[id].sfmAgent.forces;\n"
                "  };",

                "  sfm::Forces getAgentForces(int id)\n"
                "  {\n"
                "    return agents_[id].sfmAgent.forces;\n"
                "  };\n"
                "\n"
                "  /**\n"
                "   * @brief Set the scale factor for the robot's social force contribution.\n"
                "   *        0.0 = robot has no social force on agents,\n"
                "   *        1.0 = robot treated the same as a human agent.\n"
                "   */\n"
                "  void setRobotForceScale(double scale) { robot_force_scale_ = scale; }"
            ),
            # Add robot_force_scale_ member variable
            (
                "  double time_step_secs_;\n"
                "  rclcpp::Time prev_time_;",

                "  double time_step_secs_;\n"
                "  rclcpp::Time prev_time_;\n"
                "  double robot_force_scale_ = 1.0;"
            ),
        ],

        # ── agent_manager.cpp ──────────────────────────────────────────────
        "agent_manager.cpp": [
            # BEH_REGULAR case: scale robot force
            (
                "      case hunav_msgs::msg::AgentBehavior::BEH_REGULAR:\n"
                "        // We add the robot as another human agent.\n"
                "        otherAgents.push_back(robot_.sfmAgent);\n"
                "        sfm::SFM.computeForces(agents_[id].sfmAgent, otherAgents);\n"
                "        break;",

                "      case hunav_msgs::msg::AgentBehavior::BEH_REGULAR:\n"
                "        // We add the robot as another human agent, with scaled social force.\n"
                "        {\n"
                "          sfm::Agent scaledRobot = robot_.sfmAgent;\n"
                "          scaledRobot.params.forceFactorSocial *= robot_force_scale_;\n"
                "          otherAgents.push_back(scaledRobot);\n"
                "        }\n"
                "        sfm::SFM.computeForces(agents_[id].sfmAgent, otherAgents);\n"
                "        break;"
            ),
            # else branch: scale robot force
            (
                "  else\n"
                "  {\n"
                "    otherAgents.push_back(robot_.sfmAgent);\n"
                "    sfm::SFM.computeForces(agents_[id].sfmAgent, otherAgents);\n"
                "  }",

                "  else\n"
                "  {\n"
                "    sfm::Agent scaledRobot = robot_.sfmAgent;\n"
                "    scaledRobot.params.forceFactorSocial *= robot_force_scale_;\n"
                "    otherAgents.push_back(scaledRobot);\n"
                "    sfm::SFM.computeForces(agents_[id].sfmAgent, otherAgents);\n"
                "  }"
            ),
        ],

        # ── bt_functions.hpp ───────────────────────────────────────────────
        "bt_functions.hpp": [
            (
                "  sfm::Forces getAgentForces(int id) {\n"
                "    return agent_manager_.getAgentForces(id);\n"
                "  };",

                "  sfm::Forces getAgentForces(int id) {\n"
                "    return agent_manager_.getAgentForces(id);\n"
                "  };\n"
                "\n"
                "  void setRobotForceScale(double scale) {\n"
                "    agent_manager_.setRobotForceScale(scale);\n"
                "  };"
            ),
        ],

        # ── bt_node.hpp ────────────────────────────────────────────────────
        "bt_node.hpp": [
            # Add Float32MultiArray include
            (
                "#include <std_msgs/msg/color_rgba.hpp>",
                "#include <std_msgs/msg/color_rgba.hpp>\n"
                "#include <std_msgs/msg/float32_multi_array.hpp>"
            ),
            # Add publisher and scale member
            (
                "  rclcpp::Publisher<people_msgs::msg::People>::SharedPtr people_publisher_;",
                "  rclcpp::Publisher<people_msgs::msg::People>::SharedPtr people_publisher_;\n"
                "  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr robot_force_pub_;\n"
                "  double robot_force_scale_ = 0.0;"
            ),
            # Add function declaration
            (
                "  sfm::Forces getAgentForces(int id) { return btfunc_.getAgentForces(id); };",
                "  sfm::Forces getAgentForces(int id) { return btfunc_.getAgentForces(id); };\n"
                "\n"
                "  /**\n"
                "   * @brief Publish per-agent robot social force magnitudes on /human_robot_forces\n"
                "   */\n"
                "  void publish_robot_forces(rclcpp::Time t,\n"
                "                            const hunav_msgs::msg::Agents::SharedPtr agents_msg);"
            ),
        ],

        # ── bt_node.cpp ────────────────────────────────────────────────────
        "bt_node.cpp": [
            # Declare parameter in constructor
            (
                "  pub_tf_     = this->declare_parameter<bool>(\"publish_tf\", true);\n"
                "  pub_forces_ = this->declare_parameter<bool>(\"publish_sfm_forces\", true);",

                "  pub_tf_     = this->declare_parameter<bool>(\"publish_tf\", true);\n"
                "  pub_forces_ = this->declare_parameter<bool>(\"publish_sfm_forces\", true);\n"
                "  robot_force_scale_ = this->declare_parameter<double>(\"robot_force_scale\", 1.0);\n"
                "  btfunc_.setRobotForceScale(robot_force_scale_);\n"
                "  RCLCPP_INFO(get_logger(), \"robot_force_scale = %.2f\", robot_force_scale_);"
            ),
            # Create publisher in constructor
            (
                "    if (pub_people_) {\n"
                "      people_publisher_ = this->create_publisher<people_msgs::msg::People>(\n"
                "        \"people\", 1);\n"
                "    }\n"
                "  }",

                "    if (pub_people_) {\n"
                "      people_publisher_ = this->create_publisher<people_msgs::msg::People>(\n"
                "        \"people\", 1);\n"
                "    }\n"
                "    robot_force_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(\n"
                "      \"human_robot_forces\", 10);\n"
                "  }"
            ),
            # Add publish_robot_forces call in computeAgentsService
            (
                "    if (pub_people_)\n"
                "      publish_people(t, ag);\n"
                "\n"
                "    double time_step_secs = (rclcpp::Time(ag->header.stamp) - prev_time_).seconds();\n"
                "    // if the time was reset, we get a negative value\n"
                "    if (time_step_secs < 0.0)\n"
                "      time_step_secs = 0.0; // 0.05\n"
                "\n"
                "    //BT::NodeStatus status = tree_tick(time_step_secs);",

                "    if (pub_people_)\n"
                "      publish_people(t, ag);\n"
                "    publish_robot_forces(t, ag);\n"
                "\n"
                "    double time_step_secs = (rclcpp::Time(ag->header.stamp) - prev_time_).seconds();\n"
                "    // if the time was reset, we get a negative value\n"
                "    if (time_step_secs < 0.0)\n"
                "      time_step_secs = 0.0; // 0.05\n"
                "\n"
                "    //BT::NodeStatus status = tree_tick(time_step_secs);"
            ),
            # Add publish_robot_forces function implementation before publish_agents_forces
            (
                "  void BTnode::publish_agents_forces(rclcpp::Time t, const hunav_msgs::msg::Agents::SharedPtr msg)",
                "  void BTnode::publish_robot_forces(rclcpp::Time t,\n"
                "                                    const hunav_msgs::msg::Agents::SharedPtr agents_msg)\n"
                "  {\n"
                "    (void)t;\n"
                "    std_msgs::msg::Float32MultiArray msg;\n"
                "    for (const auto &a : agents_msg->agents)\n"
                "    {\n"
                "      sfm::Forces frs = getAgentForces(a.id);\n"
                "      msg.data.push_back(static_cast<float>(frs.robotSocialForce.getX()));\n"
                "      msg.data.push_back(static_cast<float>(frs.robotSocialForce.getY()));\n"
                "    }\n"
                "    robot_force_pub_->publish(msg);\n"
                "  }\n"
                "\n"
                "  void BTnode::publish_agents_forces(rclcpp::Time t, const hunav_msgs::msg::Agents::SharedPtr msg)"
            ),
        ],

        # ── CMakeLists.txt ─────────────────────────────────────────────────
        "CMakeLists.txt": [
            (
                "find_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(hunav_msgs REQUIRED)",
                "find_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(std_msgs REQUIRED)\nfind_package(hunav_msgs REQUIRED)"
            ),
            (
                "ament_target_dependencies(hunav_agent_manager rclcpp hunav_msgs",
                "ament_target_dependencies(hunav_agent_manager rclcpp std_msgs hunav_msgs"
            ),
        ],

        # ── package.xml ────────────────────────────────────────────────────
        "package.xml": [
            (
                "  <depend>rclcpp</depend>\n  <depend>geometry_msgs</depend>",
                "  <depend>rclcpp</depend>\n  <depend>std_msgs</depend>\n  <depend>geometry_msgs</depend>"
            ),
        ],
    }

    print(f"Applying robot_force_scale patch to {pkg}\n")
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue
        results = patch_file(path, patches[name])
        for i, r in enumerate(results):
            status = "✓" if r in ("OK", "ALREADY_APPLIED") else "✗"
            print(f"  {status} {name} [{i+1}/{len(results)}]: {r}")

    print("\nDone! Now rebuild:")
    print(f"  cd {ws} && colcon build --packages-select hunav_agent_manager")


if __name__ == "__main__":
    main()
