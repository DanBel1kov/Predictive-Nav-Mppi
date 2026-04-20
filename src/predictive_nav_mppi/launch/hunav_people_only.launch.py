from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def _make_nodes(context, *args, **kwargs):
    hunav_params_file = LaunchConfiguration('hunav_params_file').perform(context)
    base_world = LaunchConfiguration('base_world').perform(context)
    behavior_tree_path = LaunchConfiguration('behavior_tree_path').perform(context)
    use_sim_time_raw = LaunchConfiguration('use_sim_time').perform(context)
    humans_ignore_robot_raw = LaunchConfiguration('humans_ignore_robot').perform(context)
    robot_force_scale_raw = LaunchConfiguration('robot_force_scale').perform(context)
    # LaunchConfiguration resolves to a string; hunav_* nodes declare use_sim_time as bool.
    use_sim_time = str(use_sim_time_raw).lower() in ('true', '1', 'yes', 'on')
    humans_ignore_robot = str(humans_ignore_robot_raw).lower() in ('true', '1', 'yes', 'on')
    robot_force_scale = float(robot_force_scale_raw)

    if behavior_tree_path.strip():
        bt_dir = behavior_tree_path
    else:
        bt_dir = PathJoinSubstitution(
            [FindPackageShare('hunav_extension'), 'behavior_trees']
        )

    robot_name = 'waffle'
    ignored_models = 'waffle' if humans_ignore_robot else ''

    worldgen = Node(
        package='hunav_gazebo_wrapper',
        executable='hunav_gazebo_world_generator',
        name='hunav_gazebo_world_generator',
        output='screen',
        parameters=[{
            'base_world': base_world,
            'use_gazebo_obs': True,
            'use_collision': not humans_ignore_robot,
            # Proxying compute_agents adds service overhead; lower rate prevents
            # request timeouts and preserves human-human interactions.
            'update_rate': 20.0 if humans_ignore_robot else 100.0,
            'robot_name': robot_name,
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': False,
            'navgoal_topic': 'goal_pose',
            'ignore_models': ignored_models,
            'actor_scale': 0.5,
            # Lower actors by ~1 m from the previous setup to bring feet onto the floor.
            'actor_z_offset': -0.55,
        }],
    )

    hunav_loader_node = Node(
        package='hunav_agent_manager',
        executable='hunav_loader',
        name='hunav_loader',
        output='screen',
        parameters=[
            hunav_params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    hunav_manager_node = Node(
        package='hunav_agent_manager',
        executable='hunav_agent_manager',
        name='hunav_agent_manager',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'behavior_tree_path': bt_dir,
            'robot_force_scale': robot_force_scale,
        }],
        remappings=(
            [('compute_agents', 'compute_agents_raw')]
            if humans_ignore_robot else []
        ),
    )

    # hunav_agent_manager needs get_parameters from hunav_loader; delay so service is ready
    nodes = [
        hunav_loader_node,
        TimerAction(period=1.0, actions=[worldgen]),
        TimerAction(period=3.0, actions=[hunav_manager_node]),
    ]

    if humans_ignore_robot:
        nodes.append(
            Node(
                package='predictive_nav_mppi',
                executable='compute_agents_proxy',
                name='compute_agents_proxy',
                output='screen',
                parameters=[{
                    'backend_service': 'compute_agents_raw',
                    'frontend_service': 'compute_agents',
                    'robot_mask_distance': 10000.0,
                    'robot_force_scale': robot_force_scale,
                }],
            )
        )

    return nodes

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    default_params = PathJoinSubstitution(
        [FindPackageShare('predictive_nav_mppi'), 'config', 'hunav_agents_params.yaml']
    )
    default_world = PathJoinSubstitution(
        [FindPackageShare('room_world'), 'worlds', 'room_box.sdf']
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('hunav_params_file', default_value=default_params),
        DeclareLaunchArgument('base_world', default_value=default_world),
        DeclareLaunchArgument('behavior_tree_path', default_value=''),
        DeclareLaunchArgument('humans_ignore_robot', default_value='True'),
        DeclareLaunchArgument('robot_force_scale', default_value='0.0',
                              description='Scale of robot social force on pedestrians (0.0=invisible, 1.0=full)'),
        OpaqueFunction(function=_make_nodes),
    ])
