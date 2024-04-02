#pragma once

#include "common_includes.hpp"
#include <pangolin/pangolin.h>

class Visualizer{
public:
    Visualizer() {}

    void displayPoses(const std::vector<gtsam::Pose3> &poses);
    void loadGT(std::vector<gtsam::Pose3> &_gt_poses);
    void drawGT(const std::vector<gtsam::Pose3> &_gt_poses);
    void displayPoseWithKeypoints(const std::vector<gtsam::Pose3> &poses, const std::vector<cv::Mat> &keypoints_3d_vec);

};