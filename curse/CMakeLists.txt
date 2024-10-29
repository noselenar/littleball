cmake_minimum_required(VERSION 3.10)
project(curse)
project(iolook)
project(openpic)

set(CMAKE_TOOLCHAIN_FILE "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/installed/x64-windows")

set(OpenCV_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/installed/x64-windows/share/opencv4")
set(Protobuf_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/installed/x64-windows/share/protobuf")
set(ONNXRUNTIME_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/onnxruntime")
set(TIFF_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/packages/tiff_x64-windows/share/tiff")
set(TIFF_LIBRARY "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/packages/tiff_x64-windows/lib")
set(utf8_range_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/packages/utf8-range_x64-windows/share/utf8_range")
set(absl_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/packages/abseil_x64-windows/share/absl")
set(TIFF_INCLUDE_DIR "E:/OneDrive/Selenar/OneDrive - MSFT/proj/littleball/curse/vcpkg/packages/tiff_x64-windows/include")

find_package(OpenCV REQUIRED)
#find_package(ONNXRuntime CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
#find_package(absl REQUIRED)
#find_package(utf8_range REQUIRED)
#find_package(TIFF REQUIRED)

set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")


add_executable(curse cvtest.cpp)
add_executable(iolook iolook.cpp)
add_executable(openpic openpic.cpp)

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

target_link_libraries(curse ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${CUDA_LIBRARIES})
target_link_libraries(iolook ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
target_link_libraries(openpic ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

