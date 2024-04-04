#include "frame.hpp"

// static definition
int Frame::total_frame_cnt_ = 0;

Frame::Frame() {
    id_ = total_frame_cnt_++;

    pose_ = Eigen::Isometry3d::Identity();
    relative_pose_ = Eigen::Isometry3d::Identity();
}

void Frame::setKeypoints(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors) {
    // init keypoints_
    keypoints_ = keypoints;

    // init descriptors
    descriptors_ = descriptors;

    // init 3D keypoints
    keypoints_3d_.assign(keypoints.size(), Eigen::Vector3d(0.0, 0.0, 0.0));

    // init depths_
    depths_.reserve(keypoints.size());

    // init keypoint_landmark_
    keypoint_landmark_.assign(keypoints.size(), std::pair(-1, -1));
}