#pragma once

#include "common_includes.hpp"
#include "frame.hpp"
// #include "visual_odometry.hpp"

class TestMatch {
public:
    TestMatch(int _queryIdx, int _trainIdx) {
        queryIdx = _queryIdx;
        trainIdx = _trainIdx;
    };

    int queryIdx;
    int trainIdx;
};

class VisualOdometry;   // incomplete type

class Tester {
public:
    Tester();

    void run(VisualOdometry &vo);

    void decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat intrinsic, std::vector<cv::Point2f> image0_kp_pts, std::vector<cv::Point2f> image1_kp_pts, const cv::Mat &mask, cv::Mat &R, cv::Mat &t);
    int getPositiveLandmarksCount(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask);

    void setFrameKeypoints_pt(const std::shared_ptr<Frame> &pFrame, std::vector<cv::Point2f> kp_pts);
    void setFrameKeypoints_pt(const std::shared_ptr<Frame> &pFrame, std::vector<std::vector<cv::Point2f>> kp_pts_vec);
    void setFrameMatches(const std::shared_ptr<Frame> &pFrame, const std::vector<TestMatch> &matches_with_prev_frame);

    void triangulate3(const VisualOdometry &visual_odometry,
                            cv::Mat camera_Matrix,
                            std::shared_ptr<Frame> &pPrev_frame,
                            std::shared_ptr<Frame> &pCurr_frame,
                            const std::vector<TestMatch> good_matches,
                            const cv::Mat &mask,
                            std::vector<Eigen::Vector3d> &frame_keypoints_3d);

    void drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &good_matches,
                                    // const cv::Mat &essential_mask,
                                    const cv::Mat &pose_mask,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps);
    void drawMatches(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, const std::vector<TestMatch> &good_matches);

    void reprojectLandmarks(const std::shared_ptr<Frame> &pFrame,
                            const std::vector<TestMatch> &matches,
                            const cv::Mat &mask,
                            const std::vector<Eigen::Vector3d> &landmark_points_3d,
                            std::vector<cv::Point2f> &prev_projected_pts,
                            std::vector<cv::Point2f> &curr_projected_pts);
    double calcReprojectionError(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &matches,
                                    const cv::Mat &mask,
                                    const std::vector<Eigen::Vector3d> &landmark_points_3d);

    double estimateScale(VisualOdometry &vo, const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask);

    std::vector<cv::Point2f> findMultipleCheckerboards(const cv::Mat &image, const cv::Size &patternSize, int nCheckerboards);
    std::vector<std::vector<cv::Point2f>> findMultipleCheckerboards(const cv::Mat &image, const std::vector<cv::Size> &patternSizes, int nCheckerboardsEach);

    std::vector<cv::Point2f> manual_kp_frame0_;
    std::vector<cv::Point2f> manual_kp_frame1_;
    std::vector<cv::Point2f> manual_kp_frame2_;
    std::vector<cv::Point2f> manual_kp_frame3_;
    std::vector<cv::Point2f> manual_kp_frame4_;
    std::vector<std::vector<cv::Point2f>> manual_kps_vec_;

    std::vector<TestMatch> manual_match_0_1_;
    std::vector<TestMatch> manual_match_1_2_;
    std::vector<TestMatch> manual_match_2_3_;
    std::vector<TestMatch> manual_match_3_4_;
    std::vector<std::vector<TestMatch>> manual_matches_vec_;
};