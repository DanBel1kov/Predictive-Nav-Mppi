from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --- args ---
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')

    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    publish_initial_pose = LaunchConfiguration('publish_initial_pose')

    pkg_room = FindPackageShare('room_world')
    pkg_nav2 = FindPackageShare('nav2_bringup')

    world = PathJoinSubstitution([pkg_room, 'worlds', 'room_box.sdf'])

    # --- env fixes (у тебя это руками в терминале былоS было) ---
    env = [
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', '/opt/ros/humble/share/turtlebot3_gazebo/models'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_SHM', '0'),
        # если WSL / софт-рендер нужен — оставь, иначе можно убрать
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'),
    ]

    # --- Gazebo ---
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            world
        ],
        output='screen'
    )

    # --- Spawn robot (даём газебе подняться) ---
    spawn_waffle = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                    '-entity', 'waffle',
                    '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf',
                    '-x', '0', '-y', '0', '-z', '0.05'
                ],
                output='screen'
            )
        ]
    )

    # --- robot_state_publisher ---
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

    # --- Nav2 bringup (ВАЖНО: use_rviz:=False, RViz запустим сами с твоим конфигом) ---
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
        }.items()
    )

    # --- RViz (с твоим topdown конфигом) ---
    rviz = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen'
            )
        ]
    )

    # --- Auto initial pose (чтобы AMCL перестал ныть) ---
    # Публикуем позже, когда AMCL уже подписался на /initialpose
    initpose_cmd = (
        "ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
        "\"{header: {frame_id: 'map'}, pose: {pose: {position: {x: "
        + initial_pose_x + ", y: " + initial_pose_y + ", z: 0.0}, "
        + "orientation: {z: " + initial_pose_yaw + ", w: 1.0}}, "
        + "covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, "
        + "0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.0685]}}}\""
    )

    publish_initpose = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['bash', '-lc', f"if [ '{publish_initial_pose}' = 'True' ]; then {initpose_cmd}; fi"],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('map', default_value='/home/$USER/predictive-nav-mppi/maps/room_map.yaml'),
        DeclareLaunchArgument('params_file', default_value='/home/$USER/predictive-nav-mppi/nav2_configs/nav2_params_full.yaml'),
        DeclareLaunchArgument('rviz_config', default_value='/home/$USER/predictive-nav-mppi/rviz/nav2_topdown.rviz'),

        DeclareLaunchArgument('publish_initial_pose', default_value='True'),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),

        *env,
        gazebo,
        spawn_waffle,
        rsp,
        nav2,
        rviz,
        publish_initpose,
    ])
