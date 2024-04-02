#include "frame.hpp"

// static definition
int Frame::id_ = -1;
std::shared_ptr<Camera> Frame::pCamera_ = std::make_shared<Camera>();

void Frame::setKeypoints(const std::vector<cv::KeyPoint> &keypoints) {
    // init keypoints_
    keypoints_ = keypoints;

    // init depths_
    depths_.reserve(keypoints.size());

    // init keypoint_landmark_
    keypoint_landmark_.reserve(keypoints.size());
    keypoint_landmark_.assign(keypoints.size(), std::pair(-1, -1));
}