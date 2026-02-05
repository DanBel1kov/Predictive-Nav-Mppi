from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def _make_nodes(context, *args, **kwargs):
    hunav_params_file = LaunchConfiguration('hunav_params_file').perform(context)
    base_world = LaunchConfiguration('base_world').perform(context)
    use_sim_time_raw = LaunchConfiguration('use_sim_time').perform(context)
    # LaunchConfiguration resolves to a string; hunav_* nodes declare use_sim_time as bool.
    use_sim_time = str(use_sim_time_raw).lower() in ('true', '1', 'yes', 'on')

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
            'use_gazebo_obs': False,
            'use_collision': False,
            'update_rate': 100.0,
            'robot_name': 'waffle',
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': False,
            'navgoal_topic': 'goal_pose',
            'ignore_models': '',
            'actor_scale': 0.5,
            'actor_z_offset': -0.5,
        }],
    )

    return [
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
        ),
    ]

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
        OpaqueFunction(function=_make_nodes),
    ])
