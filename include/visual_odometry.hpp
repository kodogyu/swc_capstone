#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include <gtsam/geometry/Pose3.h>

#include "camera.hpp"
#include "frame.hpp"
#include "landmark.hpp"
#include "visualizer.hpp"
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "logger.hpp"
#include "tester.hpp"

class VisualOdometry {
public:
    VisualOdometry(std::string config_path);

    void run();
    cv::Mat readImage(int img_entry_idx);
    void triangulate(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, Eigen::Isometry3d relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
    void triangulate2(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> good_matches, const cv::Mat &mask, std::vector<Eigen::Vector3d> &frame_keypoints_3d);
    void triangulate3(cv::Mat camera_Matrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> good_matches, const cv::Mat &mask, std::vector<Eigen::Vector3d> &frame_keypoints_3d);
    double calcCovisibleLandmarkDistance(const Frame &frame, const std::vector<int> &covisible_feature_idxs);
    double estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask);
    void applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask);
    double getGTScale(std::shared_ptr<Frame> pFrame);
    void getGTScales(const std::string gt_path, bool is_kitti, int num_frames, std::vector<double> &gt_scales);

    // modules
    std::shared_ptr<Configuration> pConfig_;
    std::shared_ptr<Utils> pUtils_;
    Logger logger_;
    std::shared_ptr<Visualizer> pVisualizer_;
    std::shared_ptr<Camera> pCamera_;
    LocalOptimizer optimizer_;
    Tester tester_;
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::DescriptorMatcher> orb_matcher_;
    cv::Ptr<cv::DescriptorMatcher> sift_matcher_;

    // data containers
    std::vector<cv::Mat> keypoints_3d_vec_;
    std::vector<std::shared_ptr<Frame>> frames_;
    std::vector<std::shared_ptr<Frame>> frame_window_;
    std::vector<Eigen::Isometry3d> poses_;
    std::vector<Eigen::Isometry3d> relative_poses_;
    std::vector<double> scales_;

    // time costs
    std::vector<int64_t> feature_extraction_costs_;
    std::vector<int64_t> feature_matching_costs_;
    std::vector<int64_t> motion_estimation_costs_;
    std::vector<int64_t> triangulation_costs_;
    std::vector<int64_t> scaling_costs_;
    std::vector<int64_t> optimization_costs_;
    std::vector<int64_t> total_time_costs_;
};