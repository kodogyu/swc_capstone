#pragma once

#include "common_includes.hpp"
#include "landmark.hpp"
#include "camera.hpp"

class Frame{
public:
    Frame();
    void setKeypointsAndDescriptors(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);
    void setFrameMatches(const std::vector<cv::DMatch> &matches_with_prev_frame);

    static int total_frame_cnt_;

    // identification
    int id_;
    int frame_image_idx_;

    // frame image
    cv::Mat image_;

    // keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints_;
    std::vector<cv::Point2f> keypoints_pt_;
    cv::Mat descriptors_;

    // keypoint matches
    std::vector<int> matches_with_prev_frame_;
    std::vector<int> matches_with_next_frame_;

    // depth and 3d coordinate of keypoints
    std::vector<double> depths_;
    std::vector<Eigen::Vector3d> keypoints_3d_;

    // gtsam::Pose3 pose_;  // pose in world frame
    Eigen::Isometry3d pose_;  // pose in world frame
    Eigen::Isometry3d relative_pose_;  // relative pose between current and previous frame

    // landmark related
    std::vector<std::shared_ptr<Landmark>> landmarks_;
    std::vector<std::pair<int, int>> keypoint_landmark_;  // keypoint_landmark[keypoint_idx] = std::pair(landmarks_idx, landmark_id)

    // camera parameter
    std::shared_ptr<Camera> pCamera_;

    // other frame pointers
    std::weak_ptr<Frame> pPrevious_frame_;
    std::weak_ptr<Frame> pNext_frame_;
};