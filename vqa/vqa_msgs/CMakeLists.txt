project(vqa_msgs)

cmake_minimum_required(VERSION 3.5)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)


set(msg_files

    msg/VisualFeatures.msg
    msg/FeaturesInfo.msg



  )

  rosidl_generate_interfaces(${PROJECT_NAME}
    ${msg_files}
    DEPENDENCIES builtin_interfaces std_msgs sensor_msgs
)



ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)
ament_package()


