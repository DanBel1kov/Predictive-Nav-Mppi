import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

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
    base_world = LaunchConfiguration('world').perform(context).strip()
    if not base_world:
        return []

    world_path = os.path.join(os.path.dirname(base_world), 'generatedWorld.world')
    scaled_world_path = world_path

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

    try:
        tree = ET.parse(world_path)
        root = tree.getroot()
        world_el = root.find('world')
        if world_el is not None:
            physics_el = world_el.find('physics')
            if physics_el is None:
                physics_el = ET.SubElement(world_el, 'physics', {'type': 'ode'})

            max_step_raw = LaunchConfiguration('sim_max_step_size').perform(context).strip()
            update_rate_raw = LaunchConfiguration('sim_real_time_update_rate').perform(context).strip()
            speedup_raw = LaunchConfiguration('sim_speedup').perform(context).strip()

            try:
                sim_max_step_size = float(max_step_raw)
            except ValueError:
                sim_max_step_size = 0.0
            try:
                sim_real_time_update_rate = float(update_rate_raw)
            except ValueError:
                sim_real_time_update_rate = -1.0
            try:
                sim_speedup = float(speedup_raw)
            except ValueError:
                sim_speedup = 1.0

            max_step_el = physics_el.find('max_step_size')
            if max_step_el is None:
                max_step_el = ET.SubElement(physics_el, 'max_step_size')
            base_max_step = float(max_step_el.text or '0.001')
            if sim_max_step_size > 0.0:
                base_max_step = sim_max_step_size
                max_step_el.text = f'{base_max_step:.6f}'

            rtf_el = physics_el.find('real_time_factor')
            if rtf_el is None:
                rtf_el = ET.SubElement(physics_el, 'real_time_factor')
            if sim_speedup > 1.0:
                rtf_el.text = f'{sim_speedup:.6f}'

            rtur_el = physics_el.find('real_time_update_rate')
            if rtur_el is None:
                rtur_el = ET.SubElement(physics_el, 'real_time_update_rate')
            if sim_real_time_update_rate >= 0.0:
                if sim_real_time_update_rate == 0.0:
                    rtur_el.text = '0'
                else:
                    rtur_el.text = f'{sim_real_time_update_rate:.6f}'
            elif sim_speedup > 1.0:
                rtur_el.text = str(int(round((sim_speedup / base_max_step))))

            tree.write(world_path, encoding='unicode')
    except Exception:
        pass

    scaled_world_path = world_path

    return [ExecuteProcess(
        cmd=[
            'gzserver',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            scaled_world_path
        ],
        output='screen',
    )]


def _prepare_isolated_hunav_base_world(context, *args, **kwargs):
    use_hunav = LaunchConfiguration('use_hunav').perform(context)
    if use_hunav.lower() not in ('true', '1', 'yes'):
        return []

    world_path = LaunchConfiguration('world').perform(context).strip()
    if not world_path:
        return []

    worker_hint = (
        os.environ.get('ROS_DOMAIN_ID')
        or os.environ.get('GAZEBO_MASTER_URI', '').rsplit(':', 1)[-1]
        or 'default'
    )
    tmp_dir = tempfile.mkdtemp(prefix=f'predictive_nav_mppi_world_{worker_hint}_')
    isolated_world = os.path.join(tmp_dir, os.path.basename(world_path))
    shutil.copyfile(world_path, isolated_world)

    return [
        SetLaunchConfiguration('world', isolated_world),
        LogInfo(msg=['Isolated base world: ', isolated_world]),
    ]


def _prepare_world_physics(context, *args, **kwargs):
    world_path = LaunchConfiguration('world').perform(context).strip()
    speedup_raw = LaunchConfiguration('sim_speedup').perform(context).strip()
    max_step_raw = LaunchConfiguration('sim_max_step_size').perform(context).strip()
    update_rate_raw = LaunchConfiguration('sim_real_time_update_rate').perform(context).strip()
    if not world_path:
        return []

    try:
        sim_speedup = float(speedup_raw)
    except ValueError:
        sim_speedup = 1.0
    try:
        sim_max_step_size = float(max_step_raw)
    except ValueError:
        sim_max_step_size = 0.0
    try:
        sim_real_time_update_rate = float(update_rate_raw)
    except ValueError:
        sim_real_time_update_rate = -1.0

    if sim_speedup <= 1.0 and sim_max_step_size <= 0.0 and sim_real_time_update_rate < 0.0:
        return []

    try:
        tree = ET.parse(world_path)
        root = tree.getroot()
    except Exception as exc:
        return [LogInfo(msg=[f'Failed to parse world for sim_speedup: {exc}'])]

    world_el = root.find('world')
    if world_el is None:
        return [LogInfo(msg=['Failed to find <world> in SDF; sim_speedup disabled'])]

    physics_el = world_el.find('physics')
    if physics_el is None:
        physics_el = ET.SubElement(world_el, 'physics', {'type': 'ode'})

    max_step_el = physics_el.find('max_step_size')
    if max_step_el is None:
        max_step_el = ET.SubElement(physics_el, 'max_step_size')
    max_step = float(max_step_el.text or '0.001')
    if sim_max_step_size > 0.0:
        max_step = sim_max_step_size
        max_step_el.text = f'{max_step:.6f}'

    rtf_el = physics_el.find('real_time_factor')
    if rtf_el is None:
        rtf_el = ET.SubElement(physics_el, 'real_time_factor')
    if sim_speedup > 1.0:
        rtf_el.text = f'{sim_speedup:.6f}'

    rtur_el = physics_el.find('real_time_update_rate')
    if rtur_el is None:
        rtur_el = ET.SubElement(physics_el, 'real_time_update_rate')
    if sim_real_time_update_rate >= 0.0:
        if sim_real_time_update_rate == 0.0:
            rtur_el.text = '0'
        else:
            rtur_el.text = f'{sim_real_time_update_rate:.6f}'
    elif sim_speedup > 1.0:
        rtur_el.text = str(int(round((sim_speedup / max_step))))

    fd, tmp_path = tempfile.mkstemp(prefix='predictive_nav_mppi_world_', suffix='.sdf')
    os.close(fd)
    tree.write(tmp_path, encoding='unicode')

    logs = [
        SetLaunchConfiguration('world', tmp_path),
    ]
    if sim_speedup > 1.0:
        logs.append(LogInfo(msg=['Sim speedup target: ', f'{sim_speedup:.2f}x']))
    if sim_max_step_size > 0.0:
        logs.append(LogInfo(msg=['Sim max_step_size: ', f'{sim_max_step_size:.6f}']))
    if sim_real_time_update_rate >= 0.0:
        logs.append(LogInfo(msg=['Sim real_time_update_rate: ', f'{sim_real_time_update_rate:.6f}']))
    logs.append(LogInfo(msg=['World physics override: ', tmp_path]))
    return logs


def _resolve_scenario_assets(context, *args, **kwargs):
    scenario = LaunchConfiguration('scenario').perform(context).strip().lower()
    world_override = LaunchConfiguration('world').perform(context).strip()
    map_override = LaunchConfiguration('map').perform(context).strip()
    hunav_override = LaunchConfiguration('hunav_params_file').perform(context).strip()
    bt_override = LaunchConfiguration('behavior_tree_path').perform(context).strip()
    params_override = LaunchConfiguration('params_file').perform(context).strip()
    mppi_mode = LaunchConfiguration('mppi_mode').perform(context).strip().lower()
    if mppi_mode in ('standart', 'std'):
        mppi_mode = 'standard'

    if get_package_share_directory is None:
        return []

    room_world_share = get_package_share_directory('room_world')
    self_share = get_package_share_directory('predictive_nav_mppi')
    hunav_ext_share = get_package_share_directory('hunav_extension')

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
            'behavior_tree_path': os.path.join(hunav_ext_share, 'behavior_trees'),
        },
        'nonlinear_corridor': {
            'world': os.path.join(room_world_share, 'worlds', 'nonlinear_corridor.sdf'),
            'map': os.path.join(maps_dir, 'nonlinear_corridor_map.yaml'),
            'hunav_params_file': os.path.join(self_share, 'config', 'hunav_agents_nonlinear_params.yaml'),
            'behavior_tree_path': os.path.join(hunav_ext_share, 'behavior_trees', 'nonlinear_maze15'),
        },
        'labyrinth_turns': {
            'world': os.path.join(room_world_share, 'worlds', 'labyrinth_turns.sdf'),
            'map': os.path.join(maps_dir, 'labyrinth_turns_map.yaml'),
            'hunav_params_file': os.path.join(self_share, 'config', 'hunav_agents_labyrinth_params.yaml'),
            'behavior_tree_path': os.path.join(hunav_ext_share, 'behavior_trees', 'labyrinth_turns15'),
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
    final_bt_path = bt_override if bt_override else selected.get(
        'behavior_tree_path', os.path.join(hunav_ext_share, 'behavior_trees'))
    if params_override:
        final_params = params_override
    else:
        final_params = (
            standard_nav2_params if mppi_mode == 'standard' else default_nav2_params)

    return [
        SetLaunchConfiguration('world', final_world),
        SetLaunchConfiguration('map', final_map),
        SetLaunchConfiguration('hunav_params_file', final_hunav),
        SetLaunchConfiguration('behavior_tree_path', final_bt_path),
        SetLaunchConfiguration('params_file', final_params),
        LogInfo(msg=['Scenario: ', scenario]),
        LogInfo(msg=['MPPI mode: ', mppi_mode]),
        LogInfo(msg=['World: ', final_world]),
        LogInfo(msg=['Map: ', final_map]),
        LogInfo(msg=['Nav2 params: ', final_params]),
        LogInfo(msg=['HuNav params: ', final_hunav]),
        LogInfo(msg=['HuNav BT path: ', final_bt_path]),
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
        # Explicitly forward ROS_DOMAIN_ID and GAZEBO_MASTER_URI so that all
        # child processes (gzserver ROS2 plugins, hunav nodes, Nav2) are on
        # the correct domain when running parallel workers.
        SetEnvironmentVariable(
            'ROS_DOMAIN_ID',
            EnvironmentVariable('ROS_DOMAIN_ID', default_value='0'),
        ),
        SetEnvironmentVariable(
            'GAZEBO_MASTER_URI',
            EnvironmentVariable('GAZEBO_MASTER_URI', default_value='http://localhost:11345'),
        ),
    ]

    # Use gzserver by default. In WSL / headless setups gzclient often crashes due to OpenGL.
    gzserver = ExecuteProcess(
        condition=UnlessCondition(LaunchConfiguration('use_hunav')),
        cmd=[
            'gzserver',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            world
        ],
        output='screen',
    )


    gzclient = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('gui')),
        cmd=['gzclient'],
        output='screen',
    )

    spawn_waffle = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', 'waffle',
                '-file', LaunchConfiguration('robot_sdf'),
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
        condition=IfCondition(LaunchConfiguration('gui')),
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
            'behavior_tree_path': LaunchConfiguration('behavior_tree_path'),
            'humans_ignore_robot': LaunchConfiguration('humans_ignore_robot'),
            'robot_force_scale': LaunchConfiguration('robot_force_scale'),
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
    residual_model_weights = LaunchConfiguration(
        'residual_model_weights',
        default='/home/danbel1kov/predictive-nav-mppi/models/residual_predictor/best_residual_model.pt',
    )
    residual_alpha = LaunchConfiguration('residual_alpha', default='0.3')
    residual_smoothing_beta = LaunchConfiguration('residual_smoothing_beta', default='0.8')
    residual_clip_norm = LaunchConfiguration('residual_clip_norm', default='0.35')
    residual_turn_gate_enable = LaunchConfiguration('residual_turn_gate_enable', default='True')
    residual_turn_gate_tau = LaunchConfiguration('residual_turn_gate_tau', default='0.1')
    residual_turn_gate_alpha = LaunchConfiguration('residual_turn_gate_alpha', default='30.0')
    model_flip_forward_axis = LaunchConfiguration('model_flip_forward_axis', default='False')
    social_vae_repo_path = LaunchConfiguration('social_vae_repo_path', default='')
    social_vae_ckpt_path = LaunchConfiguration(
        'social_vae_ckpt_path',
        default='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/vae_hotel',
    )
    social_vae_config_path = LaunchConfiguration('social_vae_config_path', default='')
    social_vae_device = LaunchConfiguration('social_vae_device', default='')
    social_vae_pred_samples = LaunchConfiguration('social_vae_pred_samples', default='20')
    predict_robot_as_agent = LaunchConfiguration('predict_robot_as_agent', default='False')

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
            'publish_rate_hz': 5.0,
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
            'residual_model_weights': residual_model_weights,
            'residual_alpha': residual_alpha,
            'residual_smoothing_beta': residual_smoothing_beta,
            'residual_clip_norm': residual_clip_norm,
            'residual_turn_gate_enable': residual_turn_gate_enable,
            'residual_turn_gate_tau': residual_turn_gate_tau,
            'residual_turn_gate_alpha': residual_turn_gate_alpha,
            'scene_map_yaml': map_yaml,
            'scene_patch_size_m': 6.0,
            'scene_patch_pixels': 32,
            'scene_patch_align_to_heading': True,
            'model_obs_len': 8,
            'model_obs_dt': 0.4,
            'model_pred_steps_use': 5,
            'model_device': '',
            'residual_model_device': '',
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
            'predict_robot_as_agent': predict_robot_as_agent,
            'robot_frame': 'base_footprint',
            'global_frame': 'map',
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('gui', default_value='True'),
        DeclareLaunchArgument('sim_speedup', default_value='1.0',
                              description='Requested simulation speedup factor via world physics override'),
        DeclareLaunchArgument('sim_max_step_size', default_value='0.0',
                              description='Override Gazebo physics max_step_size; 0 keeps the world default'),
        DeclareLaunchArgument('sim_real_time_update_rate', default_value='-1.0',
                              description='Override Gazebo real_time_update_rate; 0 means as fast as possible, -1 keeps derived/default'),
        DeclareLaunchArgument('scenario', default_value='long_corridor',
                              description='World scenario: room_box, long_corridor, nonlinear_corridor, or labyrinth_turns'),
        DeclareLaunchArgument('mppi_mode', default_value='custom'),
        DeclareLaunchArgument('world', default_value=''),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('params_file', default_value=''),
        DeclareLaunchArgument('rviz_config', default_value='/home/danbel1kov/predictive-nav-mppi/rviz/nav2_topdown.rviz'),
        DeclareLaunchArgument('robot_sdf',
                              default_value='/home/danbel1kov/predictive-nav-mppi/assets/robot_models/turtlebot3_waffle_no_camera.sdf',
                              description='Robot SDF path used for spawn_entity; default disables camera for faster simulation'),
        DeclareLaunchArgument('publish_initial_pose', default_value='True'),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_hunav', default_value='True'),
        DeclareLaunchArgument('hunav_params_file', default_value=''),
        DeclareLaunchArgument('behavior_tree_path', default_value=''),
        DeclareLaunchArgument('humans_ignore_robot', default_value='True'),
        DeclareLaunchArgument('robot_force_scale', default_value='0.0',
                              description='Scale of robot social force on pedestrians (0.0=invisible, 1.0=full)'),
        DeclareLaunchArgument('predictor_type', default_value='kalman',
                              description='People predictor backend: kalman, model (SocialGRU), social_vae, or residual'),
        DeclareLaunchArgument('model_weights_path',
                              default_value='/home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/best_model.pt',
                              description='Path to .pt weights for model backend (used when predictor_type=model)'),
        DeclareLaunchArgument('residual_model_weights',
                              default_value='/home/danbel1kov/predictive-nav-mppi/models/residual_predictor/best_residual_model.pt',
                              description='Path to .pt weights for residual backend (used when predictor_type=residual)'),
        DeclareLaunchArgument('residual_alpha', default_value='0.3',
                              description='Scale factor for residual correction: hybrid = kalman + alpha * residual'),
        DeclareLaunchArgument('residual_smoothing_beta', default_value='0.8',
                              description='EMA smoothing for residual correction over time; 0 disables smoothing'),
        DeclareLaunchArgument('residual_clip_norm', default_value='0.35',
                              description='Max norm of residual correction per future step in meters; 0 disables clipping'),
        DeclareLaunchArgument('residual_turn_gate_enable', default_value='True',
                              description='Enable analytic turn-based gate for residual correction'),
        DeclareLaunchArgument('residual_turn_gate_tau', default_value='0.1',
                              description='Turn gate threshold in radians'),
        DeclareLaunchArgument('residual_turn_gate_alpha', default_value='30.0',
                              description='Turn gate sharpness'),
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
        OpaqueFunction(function=_prepare_world_physics),
        OpaqueFunction(function=_prepare_isolated_hunav_base_world),

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
