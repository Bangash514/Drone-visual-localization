# Date: 2026-11-02
# Change aim: build the C++ localizer with OpenCV, warnings on
# File: cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(drone_visual_localization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d flann)

add_executable(localize localize.cpp)
target_include_directories(localize PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(localize PRIVATE ${OpenCV_LIBS})
target_compile_options(localize PRIVATE -Wall -Wextra)
