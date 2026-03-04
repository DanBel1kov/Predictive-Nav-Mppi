cmake_minimum_required(VERSION 3.8)
project(nav2_controllers)

# Зависимости
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_mppi_controller REQUIRED)

include_directories(include)

# Библиотека-плагин Nav2 controller
add_library(mppi_nav2_controller SHARED
  src/mppi_nav2_controller.cpp
  src/mppi_controller.cpp
)

ament_target_dependencies(mppi_nav2_controller
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  tf2_ros
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  pluginlib
  nav2_util
)

add_library(mppi_people_critic SHARED
  src/predicted_people_critic.cpp
)

ament_target_dependencies(mppi_people_critic
  rclcpp
  rclcpp_lifecycle
  nav2_costmap_2d
  visualization_msgs
  pluginlib
  nav2_mppi_controller
)

# pluginlib XML
pluginlib_export_plugin_description_file(nav2_core nav2_controllers.xml)
pluginlib_export_plugin_description_file(nav2_mppi_controller mppi_critics.xml)


# Установка библиотеки
install(TARGETS mppi_nav2_controller mppi_people_critic
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Заголовки
install(DIRECTORY include/
  DESTINATION include
)

install(FILES nav2_controllers.xml
  DESTINATION share/${PROJECT_NAME}
)

install(FILES mppi_critics.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY params
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(mppi_nav2_controller)
ament_export_libraries(mppi_people_critic)
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  tf2_ros
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  pluginlib
  nav2_util
  nav2_mppi_controller
)

ament_package()
