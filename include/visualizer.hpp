#pragma once

#include <pangolin/pangolin.h>

#include "common_includes.hpp"
#include "frame.hpp"

class Visualizer{
public:
    Visualizer() {}
    Visualizer(int is_kitti);
    Visualizer(int is_kitti, std::string gt_path);

    void displayPoses(const std::vector<Eigen::Isometry3d> &poses, const bool display_gt, const std::string gt_path = "");
    void loadGT(std::string gt_path, std::vector<Eigen::Isometry3d> &_gt_poses);
    void drawGT(const std::vector<Eigen::Isometry3d> &_gt_poses);
    void displayPoseWithKeypoints(const std::vector<Eigen::Isometry3d> &poses, const std::vector<cv::Mat> &keypoints_3d_vec);
    void drawPositions(const std::vector<std::pair<int, int>> &positions);
    void displayFramesAndLandmarks(const std::vector<std::shared_ptr<Frame>> &frames, const bool display_gt, const std::string gt_path);

    std::string gt_path_ = "/home/kodogyu/Datasets/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt";
    bool is_kitti_ = false;
};