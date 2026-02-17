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
        ],
    },
)
