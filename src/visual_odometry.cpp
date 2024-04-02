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

void triangulate(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
void triangulate(cv::Mat cameraMatrix, std::vector<cv::Point> image0_keypoint_pts, std::vector<cv::Point> image1_keypoint_pts, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
double estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask);
void applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask);

int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;
    //**========== 0. Image load ==========**//
    if (argc != 2) {
        std::cout << "Usage: visual_odometry_example config_yaml" << std::endl;
        return 1;
    }

    std::vector<cv::Mat> keypoints_3d_vec;

    cv::FileStorage config_file(argv[1], cv::FileStorage::READ);
    int num_frames = config_file["num_frames"];
    std::vector<std::string> left_image_entries;
    std::filesystem::path left_images_dir(config_file["left_images_dir"]);
    std::filesystem::directory_iterator left_images_itr(left_images_dir);

    // this reads all image entries. Therefore length of the image entry vector may larger than the 'num_frames'
    while (left_images_itr != std::filesystem::end(left_images_itr)) {
        const std::filesystem::directory_entry left_image_entry = *left_images_itr;

        left_image_entries.push_back(left_image_entry.path());

        left_images_itr++;
    }
    // sort entry vectors
    std::sort(left_image_entries.begin(), left_image_entries.end());

    cv::Mat image0_left, image1_left;
    gtsam::Pose3 relative_pose;
    std::vector<std::shared_ptr<Frame>> frames;
    std::vector<gtsam::Pose3> poses;
    std::vector<double> scales;
    Camera camera;
    // read images
    image0_left = cv::imread(left_image_entries[0], cv::IMREAD_GRAYSCALE);
    poses.push_back(gtsam::Pose3());
    std::shared_ptr<Frame> pPrev_frame = std::make_shared<Frame>();
    Frame prev_frame = *pPrev_frame;
    pPrev_frame->image_ = image0_left;
    for (int i = 1; i < num_frames; i++) {
        image1_left = cv::imread(left_image_entries[i], cv::IMREAD_GRAYSCALE);
        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        Frame curr_frame = *pCurr_frame;
        pCurr_frame->image_ = image1_left;
        pCurr_frame->pPrevious_frame_ = pPrev_frame;
        pPrev_frame->pNext_frame_ = pCurr_frame;

        //**========== 1. Feature extraction ==========**//
        cv::Mat curr_image_descriptors;
        std::vector<cv::KeyPoint> curr_image_keypoints;
        // create orb feature extractor
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        if (i == 1) {  // first run
            cv::Mat prev_image_descriptors;
            std::vector<cv::KeyPoint> prev_image_keypoints;
            orb->detectAndCompute(pPrev_frame->image_, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
            pPrev_frame->setKeypoints(prev_image_keypoints, prev_image_descriptors);
        }
        orb->detectAndCompute(pCurr_frame->image_, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
        pCurr_frame->setKeypoints(curr_image_keypoints, curr_image_descriptors);

        //TODO matched keypoint filtering (RANSAC?)
        //**========== 2. Feature matching ==========**//
        // create a matcher
        cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

        // image0 left & image1 left (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches01_vec;
        std::vector<std::vector<cv::DMatch>> image_matches10_vec;
        double between_dist_thresh = 0.70;
        orb_matcher->knnMatch(pPrev_frame->descriptors_, pCurr_frame->descriptors_, image_matches01_vec, 2);
        orb_matcher->knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches10_vec, 2);

        std::vector<cv::DMatch> good_matches;  // good matchings
        for (int i = 0; i < image_matches01_vec.size(); i++) {
            if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * between_dist_thresh) {
                if (image_matches10_vec[image_matches01_vec[i][0].trainIdx][0].distance < image_matches10_vec[image_matches01_vec[i][0].trainIdx][1].distance) {  // 상호 유사도 체크
                    if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image_matches01_vec[i][0].trainIdx][0].trainIdx)
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
        std::vector<cv::Point> image0_kp_pts;
        std::vector<cv::Point> image1_kp_pts;
        for (auto match : good_matches) {
            image0_kp_pts.push_back(pPrev_frame->keypoints_[match.queryIdx].pt);
            image1_kp_pts.push_back(pCurr_frame->keypoints_[match.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, camera.intrinsic_.at<double>(0, 0), cv::Point2d(0, 0), cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat ransac_matches;
        cv::drawMatches(pPrev_frame->image_, pPrev_frame->keypoints_,
                        pCurr_frame->image_, pCurr_frame->keypoints_,
                        good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask);
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

        //** Motion estimation **//
        cv::Mat R, t;
        cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, camera.intrinsic_, R, t, mask);

        Eigen::Matrix3d rotation_mat;
        Eigen::Vector3d translation_mat;
        cv::cv2eigen(R, rotation_mat);
        cv::cv2eigen(t, translation_mat);
        relative_pose = gtsam::Pose3(gtsam::Rot3(rotation_mat), gtsam::Point3(translation_mat));
        Eigen::Isometry3d relative_pose_eigen;
        relative_pose_eigen.linear() = rotation_mat;
        relative_pose_eigen.translation() = translation_mat;
        pCurr_frame->relative_pose_ = relative_pose_eigen;
        poses.push_back(poses[i - 1] * relative_pose);

        //** Triangulation **//
        std::vector<Eigen::Vector3d> keypoints_3d;
        triangulate(camera.intrinsic_, pPrev_frame, pCurr_frame, good_matches, relative_pose, keypoints_3d);
        pPrev_frame->keypoints_3d_ = keypoints_3d;

        //** Scale estimation **//
        std::vector<int> scale_mask(pCurr_frame->keypoints_.size(), 0);
        double scale_ratio = estimateScale(pPrev_frame, pCurr_frame, scale_mask);
        scales.push_back(scale_ratio);
        applyScale(pCurr_frame, scale_ratio, scale_mask);

        cv::Mat keypoints_3d_mat = cv::Mat(3, pCurr_frame->keypoints_3d_.size(), CV_64F);
        for (int i = 0; i < keypoints_3d.size(); i++) {
            keypoints_3d_mat.at<double>(0, i) = pCurr_frame->keypoints_3d_[i].x();
            keypoints_3d_mat.at<double>(1, i) = pCurr_frame->keypoints_3d_[i].y();
            keypoints_3d_mat.at<double>(2, i) = pCurr_frame->keypoints_3d_[i].z();
        }
        keypoints_3d_vec.push_back(keypoints_3d_mat);

        // move on
        frames.push_back(pPrev_frame);
        pPrev_frame = pCurr_frame;
    }
    keypoints_3d_vec.push_back(cv::Mat::zeros(3, 1, CV_64F));


    //** Log **//
    std::ofstream log_file("output_logs/trajectory.csv");
    log_file << "qw,qx,qy,qz,x,y,z\n";
    for (auto pose : poses) {
        Eigen::Quaternion quaternion = pose.rotation().toQuaternion();
        Eigen::Vector3d position = pose.translation();
        log_file << quaternion.w() << "," << quaternion.x() << "," << quaternion.y() << "," << quaternion.z() << ","
                    << position.x() << "," << position.y() << "," << position.z() << "\n";
    }

    std::ofstream keypoints_file("output_logs/keypoints.csv");
    for (int i = 0; i < keypoints_3d_vec.size(); i++) {
        cv::Mat keypoints = keypoints_3d_vec[i];
        keypoints_file << "# " << i << "\n";
        for (int j = 0; j < keypoints.cols; j++) {
            keypoints_file << keypoints.at<double>(0,j) << "," << keypoints.at<double>(1,j) << "," << keypoints.at<double>(2,j) << "\n";
        }
    }

    //** Visualize **//
    Visualizer visualizer;
    // visualizer.displayPoses(poses);
    visualizer.displayPoseWithKeypoints(poses, keypoints_3d_vec);


    return 0;
}

void triangulate(cv::Mat cameraMatrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, gtsam::Pose3 relative_pose, std::vector<Eigen::Vector3d> &frame_keypoints_3d) {

    for (int i = 0; i < good_matches.size(); i++) {  // i = corresponding keypoint index
        cv::Point2f image0_kp_pt = pPrev_frame->keypoints_[good_matches[i].queryIdx].pt;
        cv::Point2f image1_kp_pt = pCurr_frame->keypoints_[good_matches[i].trainIdx].pt;

        Eigen::Vector3d versor;
        versor[0] = (image0_kp_pt.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
        versor[1] = (image0_kp_pt.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(0, 0);
        versor[2] = 1;

        double disparity = image0_kp_pt.x - image1_kp_pt.x;
        if (disparity != 0) {
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

// void triangulate(cv::Mat cameraMatrix, std::vector<cv::Point> image0_keypoint_pts, std::vector<cv::Point> image1_keypoint_pts, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d) {
//     for (int i = 0; i < image0_keypoint_pts.size(); i++) {
//         cv::Point2f image0_kp_pt = image0_keypoint_pts[i];
//         cv::Point2f image1_kp_pt = image1_keypoint_pts[i];

//         gtsam::Point3 versor;
//         versor[0] = (image0_kp_pt.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
//         versor[1] = (image0_kp_pt.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(0, 0);
//         versor[2] = 1;

//         double disparity = image0_kp_pt.x - image1_kp_pt.x;
//         if (disparity != 0) {
//             double depth = cameraMatrix.at<double>(0, 0) * relative_pose.translation().norm() / disparity;
//             gtsam::Point3 keypoint_3d = versor * depth;
//             frame_keypoints_3d.push_back(keypoint_3d);
//         }
//     }
// }

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
        if (pLandmark->observations_.find(pCurr_frame->id_) != pLandmark->observations_.end()) {
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
            pFrame->pose_ = gtsam::Pose3(gtsam::Rot3(scaled_pose.rotation()), gtsam::Point3(scaled_pose.translation()));

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


