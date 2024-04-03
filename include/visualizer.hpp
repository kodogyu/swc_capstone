#pragma once

#include "common_includes.hpp"
#include <pangolin/pangolin.h>

class Visualizer{
public:
    Visualizer() {}
    Visualizer(int is_kitti);
    Visualizer(int is_kitti, std::string gt_path);

    void displayPoses(const std::vector<gtsam::Pose3> &poses, const bool display_gt, const std::string gt_path = "");
    void loadGT(std::string gt_path, std::vector<gtsam::Pose3> &_gt_poses);
    void drawGT(const std::vector<gtsam::Pose3> &_gt_poses);
    void displayPoseWithKeypoints(const std::vector<gtsam::Pose3> &poses, const std::vector<cv::Mat> &keypoints_3d_vec);
    void drawPositions(const std::vector<std::pair<int, int>> &positions);

    std::string gt_path_ = "/home/kodogyu/Datasets/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt";
    bool is_kitti_ = false;
};