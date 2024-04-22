#include "common_includes.hpp"

class Tester {
public:
    Tester() {};

    void decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat intrinsic, std::vector<cv::Point2f> image0_kp_pts, std::vector<cv::Point2f> image1_kp_pts, const cv::Mat &mask, cv::Mat &R, cv::Mat &t);
    int getPositiveLandmarksCount(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask);

};