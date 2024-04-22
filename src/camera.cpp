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

Camera::Camera(const std::shared_ptr<Configuration> pConfig)
    : fx_(pConfig->fx_), fy_(pConfig->fy_), s_(pConfig->s_), cx_(pConfig->cx_), cy_(pConfig->cy_){
    intrinsic_.at<double>(0, 0) = pConfig->fx_;
    intrinsic_.at<double>(0, 1) = pConfig->s_;
    intrinsic_.at<double>(0, 2) = pConfig->cx_;
    intrinsic_.at<double>(1, 0) = 0;
    intrinsic_.at<double>(1, 1) = pConfig->fy_;
    intrinsic_.at<double>(1, 2) = pConfig->cy_;
    intrinsic_.at<double>(2, 0) = 0;
    intrinsic_.at<double>(2, 1) = 0;
    intrinsic_.at<double>(2, 2) = 1;

    if (pConfig->is_fisheye_) {
        xi_ = pConfig->xi_;
    }
    k1_ = pConfig->k1_;
    k2_ = pConfig->k2_;
    p1_ = pConfig->p1_;
    p2_ = pConfig->p2_;
    k3_ = pConfig->k3_;

    distortion_ = (cv::Mat_<double>(5, 1) << k1_, k2_, p1_, p2_, k3_);
}