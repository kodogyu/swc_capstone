#include "frame.hpp"

// static definition
int Frame::total_frame_cnt_ = 0;

Frame::Frame() {
    id_ = total_frame_cnt_++;

    pose_ = Eigen::Isometry3d::Identity();
    relative_pose_ = Eigen::Isometry3d::Identity();
}

void Frame::setKeypointsAndDescriptors(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors) {
    // init keypoints_
    keypoints_ = keypoints;

    // init keypoints_pt_
    for (unsigned int i = 0; i < keypoints_.size(); i++) {
        keypoints_pt_.push_back(keypoints[i].pt);
    }
    // subpixel
    cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001 );
    cv::cornerSubPix(image_, keypoints_pt_, cv::Size(5, 5), cv::Size(-1, -1), criteria);

    // init descriptors
    descriptors_ = descriptors;

    // init matches
    matches_with_prev_frame_ = std::vector<int>(keypoints.size(), -1);
    matches_with_next_frame_ = std::vector<int>(keypoints.size(), -1);

    // init 3D keypoints
    keypoints_3d_.assign(keypoints.size(), Eigen::Vector3d(0.0, 0.0, 0.0));

    // init depths_
    depths_.reserve(keypoints.size());

    // init keypoint_landmark_
    keypoint_landmark_.assign(keypoints.size(), std::pair(-1, -1));
}

void Frame::setFrameMatches(const std::vector<cv::DMatch> &matches_with_prev_frame) {
    int queryIdx, trainIdx;
    std::shared_ptr<Frame> pPrev_frame = pPrevious_frame_.lock();

    for (unsigned int i = 0; i < matches_with_prev_frame.size(); i++) {
        queryIdx = matches_with_prev_frame[i].queryIdx;
        trainIdx = matches_with_prev_frame[i].trainIdx;

        matches_with_prev_frame_[queryIdx] = trainIdx;
        pPrev_frame->matches_with_next_frame_[trainIdx] = queryIdx;
    }
}


void Frame::writeKeypoints(const std::string kp_dir) {
    std::string kp_filename = kp_dir + "frame" + std::to_string(id_) + ".txt";
    std::ofstream kp_file;
    kp_file.open(kp_filename);

    for (cv::KeyPoint kp : keypoints_) {
        kp_file << kp.angle << " " << kp.class_id << " " << kp.octave << " " << kp.pt << " " << kp.response << " " << kp.size << std::endl;
    }
    kp_file.close();

    std::cout << "number of keypoint [frame " << id_ << "] " << keypoints_.size() << ". " << kp_filename << std::endl;
}
