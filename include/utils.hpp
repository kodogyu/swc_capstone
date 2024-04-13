#pragma once

#include <sstream>
#include <fstream>

#include "common_includes.hpp"
#include "frame.hpp"
#include "config.hpp"

class Utils {
public:
    Utils() {};
    Utils(std::shared_ptr<Configuration> pConfig);

    void drawFramesLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);
    void drawReprojectedLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);
    void drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                const std::vector<cv::DMatch> &good_matches,
                                const cv::Mat &mask,
                                const std::vector<Eigen::Vector3d> &triangulated_kps);
    void drawGrid(cv::Mat &image);
    void drawKeypoints(std::shared_ptr<Frame> pFrame,
                    std::string folder,
                    std::string tail);

    void alignPoses(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, std::vector<Eigen::Isometry3d> &aligned_est_poses);
    std::vector<Eigen::Isometry3d> calcRPE(const std::vector<std::shared_ptr<Frame>> &frames);
    std::vector<Eigen::Isometry3d> calcRPE(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses);
    void calcRPE_rt(const std::vector<std::shared_ptr<Frame>> &frames, double &_rpe_rot, double &_rpe_trans);
    void calcRPE_rt(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, double &_rpe_rot, double &_rpe_trans);
    void loadGT(std::vector<Eigen::Isometry3d> &_gt_poses);
    Eigen::Isometry3d getGT(int &frame_idx);

    double calcReprojectionError(const std::vector<std::shared_ptr<Frame>> &frames);
    double calcReprojectionError(const std::shared_ptr<Frame> &pFrame,
                                const std::vector<cv::DMatch> &matches,
                                const cv::Mat &mask,
                                const std::vector<Eigen::Vector3d> &landmark_points_3d);

    void drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int target_frame_id, const int dup_count);
    void reprojectLandmarks(const std::shared_ptr<Frame> &pFrame,
                            const std::vector<cv::DMatch> &matches,
                            const cv::Mat &mask,
                            const std::vector<Eigen::Vector3d> &landmark_points_3d,
                            std::vector<cv::Point2f> &prev_projected_pts,
                            std::vector<cv::Point2f> &curr_projected_pts);

    void filterKeypoints(const cv::Size image_size, std::vector<cv::KeyPoint> &img_kps, cv::Mat &img_descriptor);

    std::shared_ptr<Configuration> pConfig_;
};