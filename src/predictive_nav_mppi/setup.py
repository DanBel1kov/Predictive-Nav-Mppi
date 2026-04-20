from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'predictive_nav_mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),  glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'publish_initial_pose = predictive_nav_mppi.publish_initial_pose:main',
            'people_kf_predictor = predictive_nav_mppi.people_kf_predictor:main',
            'people_predictor = predictive_nav_mppi.people_predictor:main',
            'compute_agents_proxy = predictive_nav_mppi.compute_agents_proxy:main',
            'benchmark_episode = predictive_nav_mppi.benchmark_episode:main',
            'benchmark_session = predictive_nav_mppi.benchmark_session:main',
            'run_benchmark = predictive_nav_mppi.run_benchmark:main',
            'record_people_dataset = predictive_nav_mppi.record_people_dataset:main',
            'benchmark_people_predictors = predictive_nav_mppi.benchmark_people_predictors:main',
            'curate_people_dataset = predictive_nav_mppi.curate_people_dataset:main',
            'run_curated_benchmark_suite = predictive_nav_mppi.run_curated_benchmark_suite:main',
            'train_residual_predictor = predictive_nav_mppi.train_residual_predictor:main',
            'inspect_scene_patch = predictive_nav_mppi.inspect_scene_patch:main',
            'reset_hunav_agents = predictive_nav_mppi.reset_hunav_agents:main',
            'parallel_benchmark = predictive_nav_mppi.parallel_benchmark_runner:main',
            'run_paired_benchmark = predictive_nav_mppi.run_paired_benchmark:main',
        ],
    },
)
