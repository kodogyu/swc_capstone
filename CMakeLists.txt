project(swc_capstone)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_BUILD_TYPE Debug)

##=========================DEPENDENCIES=========================##
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)   # -DOpenCV_DIR=/home/kodogyu/opencv/build
find_package(Sophus REQUIRED)
find_package(Pangolin 0.8 REQUIRED)

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "gtsam path: ${GTSAM_DIR}")

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

add_executable(visual_odometry src/visual_odometry.cpp)
target_link_libraries(visual_odometry ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)
