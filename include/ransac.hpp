#pragma once

#include <random>

#include "common_includes.hpp"
#include "utils.hpp"

class Ransac {
public:
    Ransac();
    Ransac(int sample_size, double inlier_prob, double threshold,
                const std::vector<cv::Point2f> &image0_kp_pts,
                const std::vector<cv::Point2f> &image1_kp_pts);
    void run();
    void runOnce();
    void getSamples(std::vector<cv::Point2f> &image0_kp_pts_samples, std::vector<cv::Point2f> &image1_kp_pts_samples);
    void getModel(const std::vector<cv::Point2f> &image0_kp_pts_samples, const std::vector<cv::Point2f> &image1_kp_pts_samples, Eigen::Matrix3d &fundamental_mat);
    int getInliers(const Eigen::Matrix3d &fundamental_mat, std::vector<int> &inlier_idxes);

public:
    int sample_size_ = 9;
    int sample_pool_size_;
    std::vector<cv::Point2f> image0_kp_pts_;
    std::vector<cv::Point2f> image1_kp_pts_;

    Eigen::MatrixXd best_model_;
    std::vector<int> best_inlier_idxes_;

    double alpha_ = 0.8;
    double inlier_prob_ = 0.999;

    int max_iterations_;
    double threshold_ = 3.0;

    int max_inlier_cnt_ = 0;
};