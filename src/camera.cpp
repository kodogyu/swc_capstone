#include "camera.hpp"

Camera::Camera() {
    intrinsic_.at<double>(0, 0) = fx_;
    intrinsic_.at<double>(0, 1) = s_;
    intrinsic_.at<double>(0, 2) = cx_;
    intrinsic_.at<double>(1, 0) = 0;
    intrinsic_.at<double>(1, 1) = fy_;
    intrinsic_.at<double>(1, 2) = cy_;
    intrinsic_.at<double>(2, 0) = 0;
    intrinsic_.at<double>(2, 1) = 0;
    intrinsic_.at<double>(2, 2) = 1;
}

Camera::Camera(double fx, double fy, double s, double cx, double cy)
    : fx_(fx), fy_(fy), s_(s), cx_(cx), cy_(cy) {
    intrinsic_.at<double>(0, 0) = fx_;
    intrinsic_.at<double>(0, 1) = s_;
    intrinsic_.at<double>(0, 2) = cx_;
    intrinsic_.at<double>(1, 0) = 0;
    intrinsic_.at<double>(1, 1) = fy_;
    intrinsic_.at<double>(1, 2) = cy_;
    intrinsic_.at<double>(2, 0) = 0;
    intrinsic_.at<double>(2, 1) = 0;
    intrinsic_.at<double>(2, 2) = 1;
}