cmake_minimum_required(VERSION 3.13)
project(eye_in_hand)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

add_executable(eye_in_hand main.cpp)
target_link_libraries(eye_in_hand ${OpenCV_LIBS})