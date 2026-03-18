import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction, OpaqueFunction, SetLaunchConfiguration, LogInfo
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable, TextSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


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


def _prepare_hunav_world(context, *args, **kwargs):
    use_hunav = LaunchConfiguration('use_hunav').perform(context)
    if use_hunav.lower() not in ('true', '1', 'yes'):
        return []
    if get_package_share_directory is None:
        return []

    world_path = os.path.join(
        get_package_share_directory('room_world'),
        'worlds',
        'generatedWorld.world',
    )
    scaled_world_path = '/tmp/predictive_nav_mppi_generatedWorld_scaled.world'

    # Wait for the generator to finish writing the world file.
    deadline = time.time() + 15.0
    text = ''
    last_mtime = None
    stable_since = None
    while time.time() < deadline:
        if os.path.exists(world_path):
            try:
                mtime = os.path.getmtime(world_path)
            except OSError:
                mtime = None
            if mtime is not None:
                if last_mtime == mtime:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= 0.5:
                        with open(world_path, 'r') as f:
                            text = f.read()
                        break
                else:
                    stable_since = None
                    last_mtime = mtime
            if not text:
                with open(world_path, 'r') as f:
                    text = f.read()
        time.sleep(0.2)

    scaled_world_path = world_path

    return [ExecuteProcess(
        cmd=[
            'gzserver', '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_state.so',
            scaled_world_path
        ],
        output='screen',
    )]


def _resolve_scenario_assets(context, *args, **kwargs):
    scenario = LaunchConfiguration('scenario').perform(context).strip().lower()
    world_override = LaunchConfiguration('world').perform(context).strip()
    map_override = LaunchConfiguration('map').perform(context).strip()
    hunav_override = LaunchConfiguration('hunav_params_file').perform(context).strip()
    params_override = LaunchConfiguration('params_file').perform(context).strip()
    mppi_mode = LaunchConfiguration('mppi_mode').perform(context).strip().lower()

    if get_package_share_directory is None:
        return []

    room_world_share = get_package_share_directory('room_world')
    self_share = get_package_share_directory('predictive_nav_mppi')

    # Works for this workspace layout:
    # <repo_root>/src/predictive_nav_mppi/launch/sim_nav2.launch.py
    # and gracefully falls back to legacy absolute defaults.
    repo_root_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    maps_dir_guess = os.path.join(repo_root_guess, 'maps')
    if os.path.isdir(maps_dir_guess):
        maps_dir = maps_dir_guess
    else:
        maps_dir = '/home/danbel1kov/predictive-nav-mppi/maps'

    scenarios = {
        'room_box': {
            'world': os.path.join(room_world_share, 'worlds', 'room_box.sdf'),
            'map': os.path.join(maps_dir, 'small_room_map.yaml'),
            'hunav_params_file': os.path.join(self_share, 'config', 'hunav_agents_params.yaml'),
        },
        'long_corridor': {
            'world': os.path.join(room_world_share, 'worlds', 'long_corridor.sdf'),
            'map': os.path.join(maps_dir, 'long_corridor_map.yaml'),
            'hunav_params_file': os.path.join(self_share, 'config', 'hunav_agents_corridor_params.yaml'),
        },
    }

    if scenario not in scenarios:
        scenario = 'room_box'
    if mppi_mode not in ('custom', 'standard'):
        mppi_mode = 'custom'

    selected = scenarios[scenario]
    default_nav2_params = os.path.join(self_share, 'config', 'nav2_params_full.yaml')
    standard_nav2_params = os.path.join(
        self_share, 'config', 'nav2_params_full_standard_mppi.yaml')
    final_world = world_override if world_override else selected['world']
    final_map = map_override if map_override else selected['map']
    final_hunav = hunav_override if hunav_override else selected['hunav_params_file']
    if params_override:
        final_params = params_override
    else:
        final_params = (
            standard_nav2_params if mppi_mode == 'standard' else default_nav2_params)

    return [
        SetLaunchConfiguration('world', final_world),
        SetLaunchConfiguration('map', final_map),
        SetLaunchConfiguration('hunav_params_file', final_hunav),
        SetLaunchConfiguration('params_file', final_params),
        LogInfo(msg=['Scenario: ', scenario]),
        LogInfo(msg=['MPPI mode: ', mppi_mode]),
        LogInfo(msg=['World: ', final_world]),
        LogInfo(msg=['Map: ', final_map]),
        LogInfo(msg=['Nav2 params: ', final_params]),
        LogInfo(msg=['HuNav params: ', final_hunav]),
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')


    pkg_nav2 = FindPackageShare('nav2_bringup')
    pkg_self = FindPackageShare('predictive_nav_mppi')

    env = [
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        # Gazebo classic needs these paths for OGRE shaders/resources.
        # Without them gzclient often crashes with "Failed to initialize scene" / missing shader lib.
        SetEnvironmentVariable(
            'GAZEBO_RESOURCE_PATH',
            [PathJoinSubstitution([pkg_self, 'media', 'models']),
             EnvironmentVariable('GAZEBO_RESOURCE_PATH', default_value=''),
             TextSubstitution(text=':/usr/share/gazebo-11')],
        ),
        SetEnvironmentVariable(
            'GAZEBO_PLUGIN_PATH',
            [EnvironmentVariable('GAZEBO_PLUGIN_PATH', default_value=''),
             TextSubstitution(text=':/usr/lib/x86_64-linux-gnu/gazebo-11/plugins')],
        ),
        SetEnvironmentVariable('OGRE_RESOURCE_PATH', '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0'),
        # Include both Turtlebot3 models and Gazebo's built-in models.
        SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            [EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
             TextSubstitution(text=':/opt/ros/humble/share/turtlebot3_gazebo/models:/usr/share/gazebo-11/models')]
        ),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_SHM', '0'),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'),
    ]

    # Use gzserver by default. In WSL / headless setups gzclient often crashes due to OpenGL.
    gzserver = ExecuteProcess(
        condition=UnlessCondition(LaunchConfiguration('use_hunav')),
        cmd=[
            'gzserver', '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_state.so',
            world
        ],
        output='screen',
    )


    gzclient = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('gui')),
        cmd=['gzclient', '--verbose'],
        output='screen',
    )

    spawn_waffle = TimerAction(
        period=5.0,
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

    hunav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_self, 'launch', 'hunav_people_only.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_hunav')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'hunav_params_file': LaunchConfiguration('hunav_params_file'),
            'base_world': world,
            'humans_ignore_robot': LaunchConfiguration('humans_ignore_robot'),
        }.items(),
    )

    hunav_delayed = TimerAction(
        period=0.2,
        actions=[hunav],
    )

    initpose = Node(
        condition=IfCondition(LaunchConfiguration('publish_initial_pose')),
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

    predictor_type = LaunchConfiguration('predictor_type', default='kalman')
    model_weights_path = LaunchConfiguration(
        'model_weights_path',
        default='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/best_model.pt',
    )
    model_flip_forward_axis = LaunchConfiguration('model_flip_forward_axis', default='False')
    social_vae_repo_path = LaunchConfiguration('social_vae_repo_path', default='')
    social_vae_ckpt_path = LaunchConfiguration(
        'social_vae_ckpt_path',
        default='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/vae_hotel',
    )
    social_vae_config_path = LaunchConfiguration('social_vae_config_path', default='')
    social_vae_device = LaunchConfiguration('social_vae_device', default='')
    social_vae_pred_samples = LaunchConfiguration('social_vae_pred_samples', default='20')

    people_predictor = Node(
        package='predictive_nav_mppi',
        executable='people_predictor',
        name='people_predictor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/people',
            'output_cloud_topic': '/predicted_people_cloud',
            'output_markers_topic': '/predicted_people_markers',
            'publish_rate_hz': 10.0,
            'pred_dt': 0.1,
            'pred_steps': 50,
            'sigma_meas': 0.08,
            'sigma_acc': 0.06,
            'sigma_p0': 0.06,
            'sigma_v0': 0.8,
            'min_dt': 0.02,
            'max_dt': 0.3,
            'track_timeout': 1.0,
            'max_people': 100,
            'publish_markers': True,
            'publish_ellipses': True,
            'ellipse_steps': 4,
            'frame_id_override': '',
            'predictor_type': predictor_type,
            'model_weights_path': model_weights_path,
            'model_obs_len': 8,
            'model_obs_dt': 0.4,
            'model_pred_steps_use': 5,
            'model_device': '',
            'model_ellipse_std': 0.08,
            'max_ellipse_scale': 0.25,
            'model_flip_forward_axis': model_flip_forward_axis,
            'social_vae_repo_path': social_vae_repo_path,
            'social_vae_ckpt_path': social_vae_ckpt_path,
            'social_vae_config_path': social_vae_config_path,
            'social_vae_ob_horizon': 8,
            'social_vae_pred_horizon': 12,
            'social_vae_ob_radius': 2.0,
            'social_vae_hidden_dim': 256,
            'social_vae_obs_dt': 0.4,
            'social_vae_pred_steps_use': 26,
            'social_vae_pred_samples': social_vae_pred_samples,
            'social_vae_max_neighbors': 16,
            'social_vae_neighbor_pad': 1e9,
            'social_vae_cov_std_floor': 0.08,
            'social_vae_device': social_vae_device,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('gui', default_value='True'),
        DeclareLaunchArgument('scenario', default_value='long_corridor',
                              description='World scenario: room_box or long_corridor'),
        DeclareLaunchArgument('mppi_mode', default_value='custom'),
        DeclareLaunchArgument('world', default_value=''),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('params_file', default_value=''),
        DeclareLaunchArgument('rviz_config', default_value='/home/danbel1kov/predictive-nav-mppi/rviz/nav2_topdown.rviz'),
        DeclareLaunchArgument('publish_initial_pose', default_value='True'),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_hunav', default_value='True'),
        DeclareLaunchArgument('hunav_params_file', default_value=''),
        DeclareLaunchArgument('humans_ignore_robot', default_value='True'),
        DeclareLaunchArgument('predictor_type', default_value='kalman',
                              description='People predictor backend: kalman, model (SocialGRU), or social_vae'),
        DeclareLaunchArgument('model_weights_path',
                              default_value='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/best_model.pt',
                              description='Path to .pt weights for model backend (used when predictor_type=model)'),
        DeclareLaunchArgument('model_flip_forward_axis', default_value='False',
                              description='If predictions point backward, set True (model trained with opposite forward axis)'),
        DeclareLaunchArgument('social_vae_repo_path', default_value='',
                              description='Path to external SocialVAE repo root (used when predictor_type=social_vae)'),
        DeclareLaunchArgument(
                              'social_vae_ckpt_path',
                              default_value='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/vae_hotel',
                              description='Path to SocialVAE checkpoint file or directory with ckpt-best'),
        DeclareLaunchArgument('social_vae_config_path', default_value='',
                              description='Optional path to SocialVAE config .py (e.g. config/hotel.py)'),
        DeclareLaunchArgument('social_vae_device', default_value='',
                              description='SocialVAE device: "" auto, or cpu/cuda'),
        DeclareLaunchArgument('social_vae_pred_samples', default_value='20',
                              description='Number of SocialVAE sampled futures per person'),

        OpaqueFunction(function=_resolve_scenario_assets),

        *env,
        gzserver,
        TimerAction(
            period=2.0,
            actions=[OpaqueFunction(function=_prepare_hunav_world)],
        ),
        gzclient,
        spawn_waffle,
        rsp,
        nav2_delayed,
        rviz_delayed,
        hunav_delayed,
        people_predictor,
        initpose,
    ])
