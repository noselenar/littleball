cmake_minimum_required(VERSION 3.10)
project(curse)

# 查找 OpenCV
set(OpenCV_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/installed/x64-windows/share/opencv4")
set(CMAKE_PREFIX_PATH "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/installed/x64-windows")

find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(curse cvtest.cpp)

# 链接 OpenCV 库
target_link_libraries(curse ${OpenCV_LIBS})

# 可选：设置包含目录（通常不需要，因为 find_package 会处理）
include_directories(${OpenCV_INCLUDE_DIRS})
