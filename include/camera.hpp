#pragma once

#include "common_includes.hpp"

class Camera {
public:
    Camera();
    Camera(double fx, double fy, double s, double cx, double cy);

    double fx_ = 788.629315;
    double fy_ = 786.382230;
    double s_ = 0.0;
    double cx_ = 687.158398;
    double cy_ = 317.752196;
    cv::Mat intrinsic_ = cv::Mat(3, 3, CV_64F);  // camera intrinsic parameter
};