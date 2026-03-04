from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def _make_nodes(context, *args, **kwargs):
    hunav_params_file = LaunchConfiguration('hunav_params_file').perform(context)
    base_world = LaunchConfiguration('base_world').perform(context)
    use_sim_time_raw = LaunchConfiguration('use_sim_time').perform(context)
    humans_ignore_robot_raw = LaunchConfiguration('humans_ignore_robot').perform(context)
    # LaunchConfiguration resolves to a string; hunav_* nodes declare use_sim_time as bool.
    use_sim_time = str(use_sim_time_raw).lower() in ('true', '1', 'yes', 'on')
    humans_ignore_robot = str(humans_ignore_robot_raw).lower() in ('true', '1', 'yes', 'on')

    bt_dir = PathJoinSubstitution(
        [FindPackageShare('hunav_extension'), 'behavior_trees']
    )

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
            'robot_name': 'waffle',
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': False,
            'navgoal_topic': 'goal_pose',
            'ignore_models': 'waffle' if humans_ignore_robot else '',
            'actor_scale': 0.5,
            'actor_z_offset': -0.5,
        }],
    )

    nodes = [
        Node(
            # In hunav_sim (Humble), hunav_loader is installed as an executable
            # inside the hunav_agent_manager package (lib/hunav_agent_manager/hunav_loader).
            package='hunav_agent_manager',
            executable='hunav_loader',
            name='hunav_loader',
            output='screen',
            parameters=[
                hunav_params_file,
                {'use_sim_time': use_sim_time},
            ],
        ),
        TimerAction(period=1.0, actions=[worldgen]),
        Node(
            package='hunav_agent_manager',
            executable='hunav_agent_manager',
            name='hunav_agent_manager',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'behavior_tree_path': bt_dir,
            }],
            remappings=(
                [('compute_agents', 'compute_agents_raw')]
                if humans_ignore_robot else []
            ),
        ),
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
        DeclareLaunchArgument('humans_ignore_robot', default_value='True'),
        OpaqueFunction(function=_make_nodes),
    ])
