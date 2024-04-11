#pragma once

#include <pangolin/pangolin.h>

#include "common_includes.hpp"
#include "frame.hpp"
#include "config.hpp"
#include "utils.hpp"

class Visualizer{
public:
    Visualizer() {};
    Visualizer(std::shared_ptr<Configuration> pConfig, std::shared_ptr<Utils> pUtils);

    void displayPoses(const std::vector<Eigen::Isometry3d> &poses);
    void drawGT(const std::vector<Eigen::Isometry3d> &_gt_poses);
    void displayPoseWithKeypoints(const std::vector<Eigen::Isometry3d> &poses, const std::vector<cv::Mat> &keypoints_3d_vec);
    void drawPositions(const std::vector<std::pair<int, int>> &positions);
    void displayFramesAndLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);

    std::shared_ptr<Configuration> pConfig_;
    std::shared_ptr<Utils> pUtils_;
};