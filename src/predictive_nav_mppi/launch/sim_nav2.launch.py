from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction, OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _publish_initial_pose(context, *args, **kwargs):
    publish = LaunchConfiguration('publish_initial_pose').perform(context)
    if publish.lower() not in ('true', '1', 'yes'):
        return []

    x = LaunchConfiguration('initial_pose_x').perform(context)
    y = LaunchConfiguration('initial_pose_y').perform(context)
    yaw = LaunchConfiguration('initial_pose_yaw').perform(context)

    msg = (
        "{header: {frame_id: 'map'}, "
        "pose: {pose: {position: {x: " + x + ", y: " + y + ", z: 0.0}, "
        "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, "
        "covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, "
        "0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.0685]}}"
    )

    return [ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/initialpose', 'geometry_msgs/msg/PoseWithCovarianceStamped', msg
        ],
        output='screen'
    )]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')


    pkg_room = FindPackageShare('room_world')
    pkg_nav2 = FindPackageShare('nav2_bringup')

    world = PathJoinSubstitution([pkg_room, 'worlds', 'room_box.sdf'])

    env = [
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', '/opt/ros/humble/share/turtlebot3_gazebo/models'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_SHM', '0'),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'),
    ]

    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            world
        ],
        output='screen'
    )

    spawn_waffle = TimerAction(
        period=2.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', 'waffle',
                '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf',
                '-x', '0', '-y', '0', '-z', '0.05'
            ],
            output='screen'
        )]
    )

    # robot_state_publisher
    urdf_path = '/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_waffle.urdf'
    with open(urdf_path, 'r') as f:
        robot_desc = f.read().replace('${namespace}', '')

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc
        }]
    )

    # Nav2 bringup + RViz в одном месте (не запускаем rviz руками отдельно)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav2, 'launch', 'bringup_launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'slam': 'False',
            'map': map_yaml,
            'params_file': params_file,
            'use_rviz': 'False',
            'rviz_config': rviz_config,
        }.items()
    )

    nav2_delayed = TimerAction(
        period=6.0,
        actions=[nav2]
    )

    rviz_delayed  = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen'
            )
        ]
    )

    initpose = Node(
        package='predictive_nav_mppi',
        executable='publish_initial_pose',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'x': initial_pose_x,
            'y': initial_pose_y,
            'yaw': initial_pose_yaw,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('map', default_value='/home/danbel1kov/predictive-nav-mppi/maps/room_map.yaml'),
        DeclareLaunchArgument('params_file', default_value='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/config/nav2_params_full.yaml'),
        DeclareLaunchArgument('rviz_config', default_value='/home/danbel1kov/predictive-nav-mppi/rviz/nav2_topdown.rviz'),
        DeclareLaunchArgument('publish_initial_pose', default_value='True'),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),

        *env,
        gazebo,
        spawn_waffle,
        rsp,
        nav2_delayed,
        rviz_delayed,
        initpose,
    ])
