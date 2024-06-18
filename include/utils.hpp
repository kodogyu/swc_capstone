#pragma once

#include <sstream>
#include <fstream>

#include "common_includes.hpp"
#include "frame.hpp"
#include "config.hpp"
#include "ransac.hpp"

class Utils {
public:
    Utils() {};
    Utils(std::shared_ptr<Configuration> pConfig);

    void drawFramesLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);
    void drawKeypoints(const std::shared_ptr<Frame> &pFrame);
    void drawReprojectedLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);
    void drawReprojectedKeypoints3D(const std::shared_ptr<Frame> &pFrame,
                                const std::vector<cv::DMatch> &good_matches,
                                const cv::Mat &pose_mask,
                                const std::vector<Eigen::Vector3d> &triangulated_kps);
    void drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<cv::DMatch> &good_matches);
    void drawCvReprojectedLandmarks(const std::shared_ptr<Frame> &pPrev_frame,
                                    const std::vector<cv::Point2f> &image0_kp_pts,
                                    const std::shared_ptr<Frame> &pCurr_frame,
                                    const std::vector<cv::Point2f> &image1_kp_pts,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps,
                                    const cv::Mat &pose_mask);
    void drawGrid(cv::Mat &image);
    void drawKeypoints(std::shared_ptr<Frame> pFrame,
                    std::string folder,
                    std::string tail);
    void drawKeypoints(std::shared_ptr<Frame> pFrame,
                        const std::vector<cv::Point2f> &kp_pts,
                        std::string folder,
                        std::string tail);
    void drawMatches(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> &good_matches);

    void alignPoses(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, std::vector<Eigen::Isometry3d> &aligned_est_poses);
    std::vector<Eigen::Isometry3d> calcRPE(const std::vector<std::shared_ptr<Frame>> &frames);
    std::vector<Eigen::Isometry3d> calcRPE(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses);
    void calcRPE_rt(const std::vector<std::shared_ptr<Frame>> &frames, double &_rpe_rot, double &_rpe_trans);
    void calcRPE_rt(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, double &_rpe_rot, double &_rpe_trans);
    void loadGT(std::vector<Eigen::Isometry3d> &_gt_poses);
    Eigen::Isometry3d getGT(const int frame_idx);

    double calcReprojectionError(const std::vector<std::shared_ptr<Frame>> &frames);
    double calcReprojectionError(const std::shared_ptr<Frame> &pFrame);
    double calcReprojectionError(const std::shared_ptr<Frame> &pFrame,
                                const std::vector<cv::DMatch> &matches,
                                const cv::Mat &mask,
                                const std::vector<Eigen::Vector3d> &landmark_points_3d);

    void drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int target_frame_id, const int dup_count);
    void reproject3DPoints(const std::shared_ptr<Frame> &pFrame,
                            const std::vector<cv::DMatch> &matches,
                            const cv::Mat &mask,
                            const std::vector<Eigen::Vector3d> &points_3d,
                            std::vector<cv::Point2f> &prev_projected_pts,
                            std::vector<cv::Point2f> &curr_projected_pts);

    void filterKeypoints(std::shared_ptr<Frame> &pFrame);
    void filterMatches(std::shared_ptr<Frame> &pFrame, std::vector<cv::DMatch> &matches);

    static Eigen::Matrix3d findFundamentalMat(const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts);
    static Eigen::Matrix3d findFundamentalMat(const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts,
                                        cv::Mat &mask, double inlier_prob, double ransac_threshold);
    static Eigen::Matrix3d findEssentialMat(const cv::Mat &intrinsic, const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts);
    static Eigen::Matrix3d findEssentialMat(const cv::Mat &intrinsic, const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts,
                                        cv::Mat &mask, double inlier_prob, double ransac_threshold);

    void recoverPose(const cv::Mat &intrinsic,
                        const cv::Mat &essential_mat,
                        const std::vector<cv::Point2f> &image0_kp_pts,
                        const std::vector<cv::Point2f> &image1_kp_pts,
                        Eigen::Isometry3d &relative_pose,
                        cv::Mat &mask);
    int chiralityCheck(const cv::Mat &intrinsic,
                    const std::vector<cv::Point2f> &image0_kp_pts,
                    const std::vector<cv::Point2f> &image1_kp_pts,
                    const Eigen::Isometry3d &cam1_pose,
                    cv::Mat &mask);
    void decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

    int countMask(const cv::Mat &mask);

    void cv_triangulatePoints(const std::shared_ptr<Frame>& pPrev_frame, const std::vector<cv::Point2f> &prev_kp_pts,
                                const std::shared_ptr<Frame>& pCurr_frame, const std::vector<cv::Point2f> &curr_kp_pts,
                                const std::vector<cv::DMatch> &good_matches, std::vector<Eigen::Vector3d> &keypoints_3d);
    void triangulateKeyPoints(const std::shared_ptr<Frame> &pFrame,
                                        std::vector<cv::Point2f> img0_kp_pts,
                                        std::vector<cv::Point2f> img1_kp_pts,
                                        std::vector<Eigen::Vector3d> &triangulated_kps);
    void triangulateKeyPoint(std::shared_ptr<Frame> &pFrame,
                                        cv::Point2f img0_kp_pt,
                                        cv::Point2f img1_kp_pt,
                                        Eigen::Vector3d &triangulated_kp);

    void fisheyeProcessing(std::vector<cv::KeyPoint> &keypoints);

public:
    std::shared_ptr<Configuration> pConfig_;
};