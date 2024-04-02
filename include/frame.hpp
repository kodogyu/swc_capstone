#pragma once

#include "common_includes.hpp"
#include "landmark.hpp"
#include "camera.hpp"

class Frame{
public:
    Frame() {id_ = total_frame_cnt_++;};

    static int total_frame_cnt_;

    int id_;
    cv::Mat image_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<double> depths_;
    std::vector<Eigen::Vector3d> keypoints_3d_;
    gtsam::Pose3 pose_;  // pose in world frame
    Eigen::Isometry3d relative_pose_;  // relative pose between current and previous frame

    // landmark related
    std::vector<std::shared_ptr<Landmark>> landmarks_;
    std::vector<std::pair<int, int>> keypoint_landmark_;  // keypoint_landmark[keypoint_idx] = std::pair(landmarks_idx, landmark_id)

    // camera parameter
    static std::shared_ptr<Camera> pCamera_;

    // other frame pointers
    std::weak_ptr<Frame> pPrevious_frame_;
    std::weak_ptr<Frame> pNext_frame_;

    void setKeypoints(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);
};