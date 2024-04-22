#pragma once

#include "common_includes.hpp"
#include "config.hpp"

class Camera {
public:
    Camera();
    Camera(double fx, double fy, double s, double cx, double cy);
    Camera(const std::shared_ptr<Configuration> pConfig);

    // KITTI-360
    double fx_ = 788.629315;
    double fy_ = 786.382230;
    double s_ = 0.0;
    double cx_ = 687.158398;
    double cy_ = 317.752196;
    double xi_ = 0;
    double k1_ = 0, k2_ = 0, p1_ = 0, p2_ = 0, k3_ = 0;

    cv::Mat intrinsic_ = cv::Mat(3, 3, CV_64F);  // camera intrinsic parameter
    cv::Mat distortion_ = cv::Mat();  // camera distortion parameter (fisheye camera)
};