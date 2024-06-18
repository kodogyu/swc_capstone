project(swc_capstone)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

##=========================DEPENDENCIES=========================##
find_package(Eigen3 3.3 REQUIRED)   # -DEigen3_DIR=/usr/lib/cmake/eigen3 (3.3.7), /usr/local/share/eigen3/cmake (3.4)
find_package(GTSAM REQUIRED)    # -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM
find_package(OpenCV 4 REQUIRED)   # -DOpenCV_DIR=/home/kodogyu/opencv/build
find_package(Sophus REQUIRED)
find_package(Pangolin 0.8 REQUIRED)

message(STATUS "eigen version: ${Eigen3_VERSION}")
message(STATUS "eigen path: ${Eigen3_DIR}")
message(STATUS "gtsam version: ${GTSAM_VERSION}")
message(STATUS "gtsam path: ${GTSAM_DIR}")
message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv path: ${OpenCV_DIR}")
message(STATUS "opencv include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "pangolin path: ${Pangolin_DIR}")
message(STATUS "pangolin include path: ${Pangolin_INCLUDE_DIRS}")

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(visual_odometry
    src/camera.cpp
    src/config.cpp
    src/frame.cpp
    src/landmark.cpp
    src/logger.cpp
    src/main.cpp
    src/optimizer.cpp
    src/ransac.cpp
    src/tester.cpp
    src/timer.cpp
    src/utils.cpp
    src/visual_odometry.cpp
    src/visualizer.cpp)
target_link_libraries(visual_odometry ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)
target_compile_definitions(visual_odometry PUBLIC OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT=1)

add_executable(view_gt
src/visualizer.cpp
src/view_gt.cpp)
target_link_libraries(view_gt ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)

add_executable(view_positions
src/visualizer.cpp
src/view_positions.cpp)
target_link_libraries(view_positions ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)

add_executable(test_module
    src/ransac.cpp
    src/test_module.cpp
    src/utils.cpp
    src/visualizer.cpp)
target_link_libraries(test_module ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)
target_compile_definitions(test_module PUBLIC OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT=1)