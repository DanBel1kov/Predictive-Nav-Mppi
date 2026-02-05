cmake_minimum_required(VERSION 3.8)
project(hunav_extension)

find_package(ament_cmake REQUIRED)

install(DIRECTORY behavior_trees
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
