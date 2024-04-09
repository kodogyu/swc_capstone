#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include <gtsam/geometry/Pose3.h>

#include "camera.hpp"
#include "frame.hpp"
#include "landmark.hpp"
#include "visualizer.hpp"
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "logger.hpp"


void triangulate(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, Eigen::Isometry3d relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
void triangulate2(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> good_matches, const cv::Mat &mask, std::vector<Eigen::Vector3d> &frame_keypoints_3d);
double estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask);
void applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask);
double getGTScale(std::shared_ptr<Frame> pFrame);
void getGTScales(const std::string gt_path, bool is_kitti, int num_frames, std::vector<double> &gt_scales);

int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;
    if (argc != 2) {
        std::cout << "Usage: visual_odometry_example config_yaml" << std::endl;
        return 1;
    }

    //**========== Parse config file ==========**//
    Configuration config(argv[1]);
    config.parse();

    //**========== Initialize variables ==========**//
    Utils utils;
    std::shared_ptr<Camera> camera = std::make_shared<Camera>(config.fx_, config.fy_, config.s_, config.cx_, config.cy_);
    LocalOptimizer optimizer;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config.num_features_, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 25);

    std::vector<cv::Mat> keypoints_3d_vec;
    Eigen::Isometry3d relative_pose;
    std::vector<std::shared_ptr<Frame>> frames;
    std::vector<std::shared_ptr<Frame>> frame_window;
    std::vector<Eigen::Isometry3d> poses;
    std::vector<Eigen::Isometry3d> relative_poses;
    std::vector<double> scales;
    std::vector<int64_t> feature_extraction_costs;
    std::vector<int64_t> feature_matching_costs;
    std::vector<int64_t> motion_estimation_costs;
    std::vector<int64_t> triangulation_costs;
    std::vector<int64_t> scaling_costs;
    std::vector<int64_t> total_time_costs;
    cv::Mat image0_left, image1_left;

    //**========== 0. Image load ==========**//
    // read images
    image0_left = cv::imread(config.left_image_entries_[0], cv::IMREAD_GRAYSCALE);
    poses.push_back(Eigen::Isometry3d::Identity());
    std::shared_ptr<Frame> pPrev_frame = std::make_shared<Frame>();
    pPrev_frame->image_ = image0_left;
    pPrev_frame->pCamera_ = camera;
    frames.push_back(pPrev_frame);

    for (int i = 1; i < config.num_frames_; i++) {
        // start timer [total time cost]
        std::chrono::time_point<std::chrono::steady_clock> total_time_start = std::chrono::steady_clock::now();

        image1_left = cv::imread(config.left_image_entries_[i], cv::IMREAD_GRAYSCALE);
        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        pCurr_frame->image_ = image1_left;
        pCurr_frame->pCamera_ = camera;
        pCurr_frame->pPrevious_frame_ = pPrev_frame;
        pPrev_frame->pNext_frame_ = pCurr_frame;

        //**========== 1. Feature extraction ==========**//
        // start timer [feature extraction]
        std::chrono::time_point<std::chrono::steady_clock> feature_extraction_start = std::chrono::steady_clock::now();

        cv::Mat curr_image_descriptors;
        std::vector<cv::KeyPoint> curr_image_keypoints;

        if (i == 1) {  // first run
            cv::Mat prev_image_descriptors;
            std::vector<cv::KeyPoint> prev_image_keypoints;
            orb->detectAndCompute(pPrev_frame->image_, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
            pPrev_frame->setKeypoints(prev_image_keypoints, prev_image_descriptors);
        }
        orb->detectAndCompute(pCurr_frame->image_, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
        pCurr_frame->setKeypoints(curr_image_keypoints, curr_image_descriptors);

        // end timer [feature extraction]
        std::chrono::time_point<std::chrono::steady_clock> feature_extraction_end = std::chrono::steady_clock::now();
        // feature extraction cost (us)
        auto feature_extraction_diff = feature_extraction_end - feature_extraction_start;
        auto feature_extraction_cost = std::chrono::duration_cast<std::chrono::microseconds>(feature_extraction_diff).count();
        feature_extraction_costs.push_back(feature_extraction_cost);

        //**========== 2. Feature matching ==========**//
        // start timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_start = std::chrono::steady_clock::now();

        // create a matcher
        cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

        // image0 & image1 (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches01_vec;
        std::vector<std::vector<cv::DMatch>> image_matches10_vec;
        // double between_dist_thresh = 0.60;
        orb_matcher->knnMatch(pPrev_frame->descriptors_, pCurr_frame->descriptors_, image_matches01_vec, 2);  // prev -> curr matches
        orb_matcher->knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches10_vec, 2);  // curr -> prev matches

        std::vector<cv::DMatch> good_matches;  // good matchings
        for (int i = 0; i < image_matches01_vec.size(); i++) {
            if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * config.des_dist_thresh_) {  // prev -> curr match에서 좋은가?
                int image1_keypoint_idx = image_matches01_vec[i][0].trainIdx;
                if (image_matches10_vec[image1_keypoint_idx][0].distance < image_matches10_vec[image1_keypoint_idx][1].distance * config.des_dist_thresh_) {  // curr -> prev match에서 좋은가?
                    if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image1_keypoint_idx][0].trainIdx)
                        good_matches.push_back(image_matches01_vec[i][0]);
                }
            }
        }

        std::cout << "original features for image" + std::to_string(i - 1) + "&" + std::to_string(i) + ": " << image_matches01_vec.size() << std::endl;
        std::cout << "good features for image" + std::to_string(i - 1) + "&" + std::to_string(i) + ": " << good_matches.size() << std::endl;
        // cv::Mat image_matches;
        // cv::drawMatches(image0_left, prev_image_keypoints, image1_left, curr_image_keypoints, good_matches, image_matches);
        // cv::imwrite("output_logs/inter_frames/frame"
        //         + std::to_string(i - 1)
        //         + "&"
        //         + std::to_string(i)
        //         + "_kp_matches(raw).png", image_matches);

        // RANSAC
        std::vector<cv::KeyPoint> image0_kps;
        std::vector<cv::KeyPoint> image1_kps;
        std::vector<cv::Point2f> image0_kp_pts;
        std::vector<cv::Point2f> image1_kp_pts;
        for (auto match : good_matches) {
            image0_kp_pts.push_back(pPrev_frame->keypoints_[match.queryIdx].pt);
            image1_kp_pts.push_back(pCurr_frame->keypoints_[match.trainIdx].pt);
            image0_kps.push_back(pPrev_frame->keypoints_[match.queryIdx]);
            image1_kps.push_back(pCurr_frame->keypoints_[match.trainIdx]);
        }

        cv::Mat mask;
        cv::Mat essential_mat = cv::findEssentialMat(image1_kp_pts, image0_kp_pts, camera->intrinsic_, cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat ransac_matches;
        cv::drawMatches(pPrev_frame->image_, pPrev_frame->keypoints_,
                        pCurr_frame->image_, pCurr_frame->keypoints_,
                        good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::putText(ransac_matches, "frame" + std::to_string(pPrev_frame->id_) + " & frame" + std::to_string(pCurr_frame->id_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::imwrite("output_logs/inter_frames/frame"
                + std::to_string(i - 1)
                + "&"
                + std::to_string(i)
                + "_kp_matches(ransac).png", ransac_matches);

        std::vector<cv::Point2f> inlier_keypoint0_pts, inlier_keypoint1_pts;
        for (int i = 0; i < good_matches.size(); i++) {
            if (mask.at<unsigned char>(0) == 1) {
                inlier_keypoint0_pts.push_back(image0_kp_pts[i]);
                inlier_keypoint1_pts.push_back(image1_kp_pts[i]);
            }
        }

        // end timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_end = std::chrono::steady_clock::now();
        // feature matching cost (us)
        auto feature_matching_diff = feature_matching_end - feature_matching_start;
        auto feature_matching_cost = std::chrono::duration_cast<std::chrono::microseconds>(feature_matching_diff).count();
        feature_matching_costs.push_back(feature_matching_cost);

        //**========== 3. Motion estimation ==========**//
        // start timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_start = std::chrono::steady_clock::now();

        cv::Mat R, t;
        cv::recoverPose(essential_mat, image1_kp_pts, image0_kp_pts, camera->intrinsic_, R, t, mask);

        Eigen::Matrix3d rotation_mat;
        Eigen::Vector3d translation_mat;
        cv::cv2eigen(R, rotation_mat);
        cv::cv2eigen(t, translation_mat);
        gtsam::Rot3 rotation = gtsam::Rot3(rotation_mat);
        gtsam::Point3 translation = gtsam::Point3(translation_mat);
        // relative_pose = gtsam::Pose3(gtsam::Rot3(rotation_mat), gtsam::Point3(translation_mat));
        relative_pose.linear() = rotation_mat;
        relative_pose.translation() = translation_mat;
        pCurr_frame->relative_pose_ = relative_pose;
        pCurr_frame->pose_ = pPrev_frame->pose_ * relative_pose;
        poses.push_back(poses[i - 1] * relative_pose);
        relative_poses.push_back(relative_pose);

        // end timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_end = std::chrono::steady_clock::now();
        // motion estimation cost (us)
        auto motion_estimation_diff = motion_estimation_end - motion_estimation_start;
        auto motion_estimation_cost = std::chrono::duration_cast<std::chrono::microseconds>(motion_estimation_diff).count();
        motion_estimation_costs.push_back(motion_estimation_cost);

        //**========== 4. Triangulation ==========**//
        // start timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_start = std::chrono::steady_clock::now();

        std::vector<Eigen::Vector3d> keypoints_3d;
        // triangulate(camera->intrinsic_, pPrev_frame, pCurr_frame, good_matches, relative_pose, keypoints_3d);
        triangulate2(camera->intrinsic_, pPrev_frame, pCurr_frame, good_matches, mask, keypoints_3d);
        pPrev_frame->keypoints_3d_ = keypoints_3d;

        // end timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_end = std::chrono::steady_clock::now();
        // motion estimation cost (us)
        auto triangulation_diff = triangulation_end - triangulation_start;
        auto triangulation_cost = std::chrono::duration_cast<std::chrono::microseconds>(triangulation_diff).count();
        triangulation_costs.push_back(triangulation_cost);

        //**========== 5. Scale estimation ==========**//
        // start timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_start = std::chrono::steady_clock::now();

        // std::vector<int> scale_mask(pCurr_frame->keypoints_.size(), 1);
        // // double scale_ratio = estimateScale(pPrev_frame, pCurr_frame, scale_mask);
        // std::cout << "get GT scale" << std::endl;
        // double scale_ratio = getGTScale(pCurr_frame);
        // scales.push_back(scale_ratio);
        // applyScale(pCurr_frame, scale_ratio, scale_mask);

        // end timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_end = std::chrono::steady_clock::now();
        // scaling time cost (us)
        auto scaling_diff = scaling_end - scaling_start;
        auto scaling_cost = std::chrono::duration_cast<std::chrono::microseconds>(scaling_diff).count();
        scaling_costs.push_back(scaling_cost);

        cv::Mat keypoints_3d_mat = cv::Mat(3, pPrev_frame->keypoints_3d_.size(), CV_64F);
        for (int i = 0; i < keypoints_3d.size(); i++) {
            keypoints_3d_mat.at<double>(0, i) = pPrev_frame->keypoints_3d_[i].x();
            keypoints_3d_mat.at<double>(1, i) = pPrev_frame->keypoints_3d_[i].y();
            keypoints_3d_mat.at<double>(2, i) = pPrev_frame->keypoints_3d_[i].z();
        }
        keypoints_3d_vec.push_back(keypoints_3d_mat);

        //**========== 6. Local optimization ==========**//
        if (config.do_optimize_) {
            if (pCurr_frame->id_ % 1 == 0) {
                frame_window.push_back(pCurr_frame);
                if (frame_window.size() > config.window_size_) {
                    frame_window.erase(frame_window.begin());
                }
                if (frame_window.size() == config.window_size_) {
                    double reprojection_error = utils.calcReprojectionError(frame_window);
                    std::cout << "calculated reprojection error: " << reprojection_error << std::endl;

                    optimizer.optimizeFrames(frame_window, config.optimizer_verbose_);
                }
            }
        }

        // move on
        frames.push_back(pCurr_frame);
        pPrev_frame = pCurr_frame;

        // end timer [total time]
        std::chrono::time_point<std::chrono::steady_clock> total_time_end = std::chrono::steady_clock::now();
        // total time cost (us)
        auto total_time_diff = total_time_end - total_time_start;
        auto total_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(total_time_diff).count();
        total_time_costs.push_back(total_time_cost);
    }
    keypoints_3d_vec.push_back(cv::Mat::zeros(3, 1, CV_64F));
    utils.drawReprojectedLandmarks(frames);
    utils.drawFramesLandmarks(frames);
    utils.drawCorrespondingFeatures(frames, 3, 2);

    //**========== Log ==========**//
    Logger logger;

    // trajectory
    logger.logTrajectory(relative_poses);
    // logger.logTrajectory(poses);

    // keypoints
    logger.logKeypoints(keypoints_3d_vec);

    // landmarks
    logger.logLandmarks(frames);

    // time cost[us]
    logger.logTimecosts(feature_extraction_costs,
                        feature_matching_costs,
                        motion_estimation_costs,
                        triangulation_costs,
                        scaling_costs,
                        total_time_costs);

    // scales
    std::vector<double> gt_scales;
    getGTScales(config.gt_path_, config.is_kitti_, config.num_frames_, gt_scales);
    logger.logScales(scales, gt_scales);

    // RPE
    double rpe_rot, rpe_trans;
    utils.calcRPE_rt(config.gt_path_, frames, rpe_rot, rpe_trans);
    logger.logRPE(rpe_rot, rpe_trans);
    std::cout << "RPEr: " << rpe_rot << std::endl;
    std::cout << "RPEt: " << rpe_trans << std::endl;

    //**========== Visualize ==========**//
    Visualizer visualizer(config.is_kitti_, config.gt_path_);
    switch(config.display_type_) {
        case 0:
            visualizer.displayPoses(poses, config.display_gt_, config.gt_path_);
            break;
        case 1:
            visualizer.displayFramesAndLandmarks(frames, config.display_gt_, config.gt_path_);
            break;
    }


    return 0;
}

void triangulate(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, Eigen::Isometry3d relative_pose, std::vector<Eigen::Vector3d> &frame_keypoints_3d) {

    for (int i = 0; i < good_matches.size(); i++) {  // i = corresponding keypoint index
        cv::Point2f image0_kp_pt = pPrev_frame->keypoints_[good_matches[i].queryIdx].pt;
        cv::Point2f image1_kp_pt = pCurr_frame->keypoints_[good_matches[i].trainIdx].pt;

        Eigen::Vector3d versor;
        versor[0] = (image0_kp_pt.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
        versor[1] = (image0_kp_pt.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(0, 0);
        versor[2] = 1;

        double disparity = image0_kp_pt.x - image1_kp_pt.x;
        if (disparity > 0) {
            bool new_landmark = true;
            // get depth
            double depth = cameraMatrix.at<double>(0, 0) * relative_pose.translation().norm() / disparity;
            pPrev_frame->depths_[good_matches[i].queryIdx] = depth;
            // get 3D point
            Eigen::Vector3d keypoint_3d = versor * depth;
            Eigen::Vector3d w_keypoint_3d = pPrev_frame->pose_ * keypoint_3d;  // keypoint coordinate in world frame
            frame_keypoints_3d.push_back(w_keypoint_3d);

            if (pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].second != -1) {
                new_landmark = false;
            }

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations_.insert({pPrev_frame->id_, good_matches[i].queryIdx});
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
                pLandmark->point_3d_ = w_keypoint_3d;

                pPrev_frame->landmarks_.push_back(pLandmark);
                pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx] = std::pair(pPrev_frame->landmarks_.size() - 1, pLandmark->id_);
                pCurr_frame->landmarks_.push_back(pLandmark);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, pLandmark->id_);
            }
            else {
                // add information to curr_frame
                std::pair<int, int> prev_frame_kp_lm = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx];
                int landmark_id = prev_frame_kp_lm.second;
                pCurr_frame->landmarks_.push_back(pPrev_frame->landmarks_[prev_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size(), landmark_id);

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrev_frame->landmarks_[prev_frame_kp_lm.first];
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
            }
        }
    }
}

void triangulate2(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> good_matches, const cv::Mat &mask, std::vector<Eigen::Vector3d> &frame_keypoints_3d) {
    Eigen::Matrix3d camera_intrinsic;
    Eigen::MatrixXd prev_proj(3, 4);
    Eigen::MatrixXd curr_proj(3, 4);

    cv::cv2eigen(pPrev_frame->pCamera_->intrinsic_, camera_intrinsic);
    prev_proj = camera_intrinsic * pPrev_frame->pose_.matrix().inverse().block<3, 4>(0, 0);
    curr_proj = camera_intrinsic * pCurr_frame->pose_.matrix().inverse().block<3, 4>(0, 0);

    for (int i = 0; i < good_matches.size(); i++) {
        if (mask.at<unsigned char>(i) == 1) {
            bool new_landmark = true;
            cv::KeyPoint prev_frame_kp = pPrev_frame->keypoints_[good_matches[i].queryIdx];
            cv::KeyPoint curr_frame_kp = pCurr_frame->keypoints_[good_matches[i].trainIdx];

            Eigen::Matrix4d A;
            A.row(0) = prev_frame_kp.pt.x * prev_proj.row(2) - prev_proj.row(0);
            A.row(1) = prev_frame_kp.pt.y * prev_proj.row(2) - prev_proj.row(1);
            A.row(2) = curr_frame_kp.pt.x * curr_proj.row(2) - curr_proj.row(0);
            A.row(3) = curr_frame_kp.pt.y * curr_proj.row(2) - curr_proj.row(1);

            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
            Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
            frame_keypoints_3d.push_back(point_3d);

            Eigen::Vector3d w_keypoint_3d = point_3d;  // keypoint coordinate in world frame
            if (pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].second != -1) {
                new_landmark = false;
            }

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations_.insert({pPrev_frame->id_, good_matches[i].queryIdx});
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
                pLandmark->point_3d_ = w_keypoint_3d;

                pPrev_frame->landmarks_.push_back(pLandmark);
                pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx] = std::pair(pPrev_frame->landmarks_.size() - 1, pLandmark->id_);
                pCurr_frame->landmarks_.push_back(pLandmark);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, pLandmark->id_);
            }
            else {
                // add information to curr_frame
                std::pair<int, int> prev_frame_kp_lm = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx];
                int landmark_id = prev_frame_kp_lm.second;
                pCurr_frame->landmarks_.push_back(pPrev_frame->landmarks_[prev_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size(), landmark_id);

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrev_frame->landmarks_[prev_frame_kp_lm.first];
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
            }
        }
    }
}


double calcCovisibleLandmarkDistance(const Frame &frame, const std::vector<int> &covisible_feature_idxs) {
    double acc_distance = 0;

    Eigen::Vector3d landmark_3d_k1, landmark_3d_k;
    std::shared_ptr<Frame> pPrev_frame = frame.pPrevious_frame_.lock();

    landmark_3d_k1 = pPrev_frame->keypoints_3d_[frame.landmarks_[frame.keypoint_landmark_[covisible_feature_idxs[0]].first]->observations_.find(pPrev_frame->id_)->second];
    for (int i = 1; i < covisible_feature_idxs.size(); i++) {  // i = landmark index
        int curr_frame_landmark_idx = frame.keypoint_landmark_[covisible_feature_idxs[i]].first;
        int prev_frame_keypoint_idx = frame.landmarks_[curr_frame_landmark_idx]->observations_.find(pPrev_frame->id_)->second;

        landmark_3d_k = pPrev_frame->keypoints_3d_[prev_frame_keypoint_idx];

        acc_distance += (landmark_3d_k - landmark_3d_k1).norm();

        landmark_3d_k1 = landmark_3d_k;
    }

    return acc_distance;
}

double estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask) {
    double scale_ratio = 1.0;

    std::vector<int> prev_frame_covisible_feature_idxs, curr_frame_covisible_feature_idxs;

    if (pPrev_frame->id_ == 0) {
        return 1.0;
    }

    for (auto pLandmark : pPrev_frame->landmarks_) {
        std::shared_ptr<Frame> pBefore_prev_frame = pPrev_frame->pPrevious_frame_.lock();
        if (pLandmark->observations_.find(pCurr_frame->id_) != pLandmark->observations_.end()  // 현재 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pPrev_frame->id_) != pLandmark->observations_.end()  // 이전 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pBefore_prev_frame->id_) != pLandmark->observations_.end()) {  // 전전 프레임에서 관측되는 landmark
            prev_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pPrev_frame->id_)->second);
            curr_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pCurr_frame->id_)->second);

            scale_mask[pLandmark->observations_.find(pCurr_frame->id_)->second] = 1;
        }
    }
    double prev_frame_landmark_distance = calcCovisibleLandmarkDistance(*pPrev_frame, prev_frame_covisible_feature_idxs);
    double curr_frame_landmark_distance = calcCovisibleLandmarkDistance(*pCurr_frame, curr_frame_covisible_feature_idxs);

    scale_ratio = curr_frame_landmark_distance / prev_frame_landmark_distance;

    return scale_ratio;
}

void applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask) {
    for (int i = 0; i < scale_mask.size(); i++) {
        if (scale_mask[i] == 1) {
            // depth
            pFrame->depths_[i] /= 1 / scale_ratio;
            // relative pose
            Eigen::Isometry3d relative_pose_old = pFrame->relative_pose_;  // copy relative pose
            Eigen::Vector3d translation = pFrame->relative_pose_.translation();
            translation /= scale_ratio;  // apply scale
            // pose
            Eigen::Isometry3d prev_pose(pFrame->pose_.matrix());
            prev_pose = Eigen::Isometry3d(pFrame->pose_.matrix()) * relative_pose_old.inverse();
            Eigen::Isometry3d scaled_pose = prev_pose * pFrame->relative_pose_;  // apply scale
            pFrame->pose_ = scaled_pose;

            // feature_3d
            Eigen::Vector3d feature_versor;
            feature_versor[0] = (pFrame->keypoints_[i].pt.x - pFrame->pCamera_->cx_) / pFrame->pCamera_->fx_;
            feature_versor[1] = (pFrame->keypoints_[i].pt.y - pFrame->pCamera_->cy_) / pFrame->pCamera_->fy_;
            feature_versor[2] = 1.0;
            Eigen::Vector3d w_feature_3d = pFrame->pose_ * (feature_versor * pFrame->depths_[i]);  // scaled 3d feature point (in world frame)
            pFrame->keypoints_3d_[i] = w_feature_3d;
            // landmark position
            int landmark_idx = pFrame->keypoint_landmark_[i].first;
            pFrame->landmarks_[landmark_idx]->point_3d_ = w_feature_3d;
        }
    }
}

double getGTScale(std::shared_ptr<Frame> pFrame) {
    std::cout << "GT scale function" << std::endl;

    if (pFrame->id_ == 1) {
        return 1.0;
    }

    std::cout << "reading file" << std::endl;
    std::ifstream gt_poses_file("/home/kodogyu/Datasets/KITTI/dataset/poses/00.txt");
    std::cout << "reading file done" << std::endl;
    std::string line;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    int curr_id = pFrame->id_;
    int prev_id = curr_id - 1;
    int pprev_id = curr_id - 2;
    Eigen::Vector3d pprev_position, prev_position, curr_position;

    for (int i = 0; i < pFrame->id_; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);

        // KITTI format
        ssline
            >> r11 >> r12 >> r13 >> t1
            >> r21 >> r22 >> r23 >> t2
            >> r31 >> r32 >> r33 >> t3;

        if (i == pprev_id) {
            pprev_position << t1, t2, t3;
            std::cout << "pprev position: " << pprev_position << std::endl;
        }
        else if (i == prev_id) {
            prev_position << t1, t2, t3;
            std::cout << "prev position: " << prev_position << std::endl;
        }
        else if (i == curr_id) {
            curr_position << t1, t2, t3;
            std::cout << "curr position: " << curr_position << std::endl;
        }
    }

    double gt_scale = (curr_position - prev_position).norm() / (prev_position - pprev_position).norm();
    return gt_scale;
}

void getGTScales(const std::string gt_path, bool is_kitti, int num_frames, std::vector<double> &gt_scales) {
    std::ifstream gt_poses_file(gt_path);
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    Eigen::Vector3d prev_position, curr_position;
    double prev_trans_length;
    double trans_length;
    double scale;

    for (int i = 0; i < num_frames; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);
        if (is_kitti) {
            ssline
                >> r11 >> r12 >> r13 >> t1
                >> r21 >> r22 >> r23 >> t2
                >> r31 >> r32 >> r33 >> t3;
        }
        else {
            ssline >> no_frame
                    >> r11 >> r12 >> r13 >> t1
                    >> r21 >> r22 >> r23 >> t2
                    >> r31 >> r32 >> r33 >> t3;
        }

        curr_position << t1, t2, t3;

        if (i == 0) {
            prev_position = curr_position;
            continue;
        }
        else if (i == 1) {
            scale = 1.0;
        }
        else {
            trans_length = (curr_position - prev_position).norm();
            scale = trans_length / prev_trans_length;
        }
        gt_scales.push_back(scale);

        prev_position = curr_position;
        prev_trans_length = trans_length;
    }
}
