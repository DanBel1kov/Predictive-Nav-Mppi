from setuptools import find_packages, setup

package_name = 'room_world'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['room_world/launch/sim_room.launch.py']),
    ('share/' + package_name + '/worlds', ['room_world/worlds/room_box.sdf']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='danbel1kov',
    maintainer_email='danbel1kov@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
