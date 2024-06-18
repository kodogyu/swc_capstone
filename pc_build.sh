echo "Configuring and building Visual Odometry in PC configuration ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM -DEigen3_DIR=/usr/local/share/eigen3/cmake
make visual_odometry -j4