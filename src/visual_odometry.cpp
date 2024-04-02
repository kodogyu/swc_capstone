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
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "camera.hpp"
#include "frame.hpp"
#include "landmark.hpp"
#include "visualizer.hpp"

void triangulate(cv::Mat cameraMatrix, Frame &prev_frame, Frame &curr_frame, std::vector<cv::DMatch> good_matches, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
void triangulate(cv::Mat cameraMatrix, std::vector<cv::Point> image0_keypoint_pts, std::vector<cv::Point> image1_keypoint_pts, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d);
double estimateScale(const Frame &prev_frame, const Frame &curr_frame, std::vector<int> &scale_mask);
void applyScale(Frame &frame, const double scale_ratio, const std::vector<int> scale_mask);

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
    std::vector<gtsam::Pose3> poses;
    std::vector<Frame> frames;
    std::vector<double> scales;
    Camera camera;
    // read images
    image0_left = cv::imread(left_image_entries[0], cv::IMREAD_GRAYSCALE);
    poses.push_back(gtsam::Pose3());
    Frame prev_frame;
    prev_frame.image_ = image0_left;
    frames.push_back(prev_frame);
    for (int i = 1; i < num_frames; i++) {
        image1_left = cv::imread(left_image_entries[i], cv::IMREAD_GRAYSCALE);
        Frame curr_frame;
        curr_frame.image_ = image1_left;

        //**========== 1. Feature extraction ==========**//
        cv::Mat image0_left_descriptors, image1_left_descriptors;
        std::vector<cv::KeyPoint> prev_image_keypoints, curr_image_keypoints;
        // create orb feature extractor
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        if (i == 1) {  // first run
            orb->detectAndCompute(prev_frame.image_, cv::Mat(), prev_image_keypoints, image0_left_descriptors);
            prev_frame.setKeypoints(prev_image_keypoints);
        }
        orb->detectAndCompute(curr_frame.image_, cv::Mat(), curr_image_keypoints, image1_left_descriptors);
        curr_frame.setKeypoints(curr_image_keypoints);

        //TODO matched keypoint filtering (RANSAC?)
        //**========== 2. Feature matching ==========**//
        // create a matcher
        cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

        // image0 left & image1 left (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches01_vec;
        std::vector<std::vector<cv::DMatch>> image_matches10_vec;
        double between_dist_thresh = 0.70;
        orb_matcher->knnMatch(image0_left_descriptors, image1_left_descriptors, image_matches01_vec, 2);
        orb_matcher->knnMatch(image1_left_descriptors, image0_left_descriptors, image_matches10_vec, 2);

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
            image0_kp_pts.push_back(prev_image_keypoints[match.queryIdx].pt);
            image1_kp_pts.push_back(curr_image_keypoints[match.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, camera.intrinsic_.at<double>(0, 0), cv::Point2d(0, 0), cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat ransac_matches;
        cv::drawMatches(prev_frame.image_, prev_image_keypoints,
                        curr_frame.image_, curr_image_keypoints,
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
        curr_frame.relative_pose_ = relative_pose_eigen;
        poses.push_back(poses[i - 1] * relative_pose);

        //** Triangulation **//
        std::vector<gtsam::Point3> keypoints_3d;
        triangulate(camera.intrinsic_, prev_frame, curr_frame, good_matches, relative_pose, keypoints_3d);
        cv::Mat keypoints_3d_mat = cv::Mat(3, keypoints_3d.size(), CV_64F);
        for (int i = 0; i < keypoints_3d.size(); i++) {
            keypoints_3d_mat.at<double>(0, i) = keypoints_3d[i].x();
            keypoints_3d_mat.at<double>(1, i) = keypoints_3d[i].y();
            keypoints_3d_mat.at<double>(2, i) = keypoints_3d[i].z();
        }
        prev_frame.keypoints_3d_ = keypoints_3d_mat;
        keypoints_3d_vec.push_back(keypoints_3d_mat);

        //** Scale estimation **//
        std::vector<int> scale_mask;
        double scale_ratio = estimateScale(prev_frame, curr_frame, scale_mask);
        scales.push_back(scale_ratio);
        applyScale(curr_frame, scale_ratio, scale_mask);


        // move on
        prev_frame = curr_frame;
    }
    keypoints_3d_vec.push_back(cv::Mat::zeros(3, 1, CV_64F));


    //** Log **//
    std::ofstream log_file("output_logs/trajectory.csv");
    log_file << "qw,qx,qy,qz,x,y,z\n";
    for (auto pose : poses) {
        gtsam::Vector quaternion = pose.rotation().quaternion();
        gtsam::Vector position = pose.translation();
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

void triangulate(cv::Mat cameraMatrix, Frame &prev_frame, Frame &curr_frame, std::vector<cv::DMatch> good_matches, gtsam::Pose3 relative_pose, std::vector<gtsam::Point3> &frame_keypoints_3d) {

    for (int i = 0; i < good_matches.size(); i++) {  // i = corresponding keypoint index
        cv::Point2f image0_kp_pt = prev_frame.keypoints_[good_matches[i].queryIdx].pt;
        cv::Point2f image1_kp_pt = curr_frame.keypoints_[good_matches[i].trainIdx].pt;

        gtsam::Point3 versor;
        versor[0] = (image0_kp_pt.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
        versor[1] = (image0_kp_pt.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(0, 0);
        versor[2] = 1;

        double disparity = image0_kp_pt.x - image1_kp_pt.x;
        if (disparity != 0) {
            bool new_landmark = true;
            // get depth
            double depth = cameraMatrix.at<double>(0, 0) * relative_pose.translation().norm() / disparity;
            prev_frame.depths_[good_matches[i].queryIdx] = depth;
            // get 3D point
            gtsam::Point3 keypoint_3d = versor * depth;
            gtsam::Point3 w_keypoint_3d = prev_frame.pose_ * keypoint_3d;  // keypoint coordinate in world frame
            frame_keypoints_3d.push_back(w_keypoint_3d);

            if (prev_frame.keypoint_landmark_[good_matches[i].queryIdx].second != -1) {
                new_landmark = false;
            }

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations.insert({prev_frame.id_, good_matches[i].queryIdx});
                pLandmark->observations.insert({curr_frame.id_, good_matches[i].trainIdx});
                pLandmark->point_3d = w_keypoint_3d;

                prev_frame.landmarks_.push_back(pLandmark);
                prev_frame.keypoint_landmark_[good_matches[i].queryIdx] = std::pair(prev_frame.landmarks_.size(), pLandmark->id);
                curr_frame.landmarks_.push_back(pLandmark);
                curr_frame.keypoint_landmark_[good_matches[i].trainIdx] = std::pair(curr_frame.landmarks_.size(), pLandmark->id);
            }
            else {
                // add information to curr_frame
                std::pair<int, int> prev_frame_kp_lm = prev_frame.keypoint_landmark_[good_matches[i].queryIdx];
                int landmark_id = prev_frame_kp_lm.second;
                curr_frame.landmarks_.push_back(prev_frame.landmarks_[prev_frame_kp_lm.first]);
                curr_frame.keypoint_landmark_[good_matches[i].trainIdx] = std::pair(curr_frame.landmarks_.size(), landmark_id);

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = prev_frame.landmarks_[prev_frame_kp_lm.first];
                pLandmark->observations.insert({curr_frame.id_, good_matches[i].trainIdx});
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
    cv::cv2eigen(frame.keypoints_3d_.rowRange(0, 3).col(0), landmark_3d_k1);
    for (int i = 1; i < covisible_feature_idxs.size(); i++) {  // i = landmark index
        int keypoint_idx = covisible_feature_idxs[i];
        cv::cv2eigen(frame.keypoints_3d_.rowRange(0, 3).col(keypoint_idx), landmark_3d_k);

        acc_distance += (landmark_3d_k - landmark_3d_k1).norm();

        landmark_3d_k1 = landmark_3d_k;
    }

    return acc_distance;
}

double estimateScale(const Frame &prev_frame, const Frame &curr_frame, std::vector<int> &scale_mask) {
    double scale_ratio = 1.0;

    std::vector<int> prev_frame_covisible_feature_idxs, curr_frame_covisible_feature_idxs;

    for (int i = 0; i < prev_frame.landmarks_.size(); i++) {
        for (int j = 0; j < curr_frame.landmarks_.size(); j++) {
            if (prev_frame.landmarks_[i]->id == prev_frame.landmarks_[j]->id) {
                std::shared_ptr pLandmark = prev_frame.landmarks_[i];

                prev_frame_covisible_feature_idxs.push_back(pLandmark->observations.find(prev_frame.id_)->second);
                curr_frame_covisible_feature_idxs.push_back(pLandmark->observations.find(curr_frame.id_)->second);
                break;
            }
        }
    }
    double prev_frame_landmark_distance = calcCovisibleLandmarkDistance(prev_frame, prev_frame_covisible_feature_idxs);
    double curr_frame_landmark_distance = calcCovisibleLandmarkDistance(curr_frame, curr_frame_covisible_feature_idxs);

    scale_ratio = curr_frame_landmark_distance / prev_frame_landmark_distance;

    scale_mask = curr_frame_covisible_feature_idxs;
    return scale_ratio;
}

void applyScale(Frame &frame, const double scale_ratio, const std::vector<int> scale_mask) {
    for (int i = 0; i < scale_mask.size(); i++) {
        if (scale_mask[i] == 1) {
            // depth
            frame.depths_[i] /= 1 / scale_ratio;
            // relative pose
            Eigen::Isometry3d relative_pose_old = frame.relative_pose_;  // copy relative pose
            Eigen::Vector3d translation = frame.relative_pose_.translation();
            translation /= scale_ratio;  // apply scale
            // pose
            Eigen::Isometry3d prev_pose(frame.pose_.matrix());
            prev_pose = Eigen::Isometry3d(frame.pose_.matrix()) * relative_pose_old.inverse();
            Eigen::Isometry3d scaled_pose = prev_pose * frame.relative_pose_;  // apply scale
            frame.pose_ = gtsam::Pose3(gtsam::Rot3(scaled_pose.rotation()), gtsam::Point3(scaled_pose.translation()));

            // feature_3d
            Eigen::Vector3d feature_versor;
            feature_versor[0] = (frame.keypoints_[i].pt.x - frame.pCamera_->cx_) / frame.pCamera_->fx_;
            feature_versor[1] = (frame.keypoints_[i].pt.y - frame.pCamera_->cy_) / frame.pCamera_->fy_;
            feature_versor[2] = 1.0;
            Eigen::Vector3d w_feature_3d = frame.pose_ * (feature_versor * frame.depths_[i]);  // scaled 3d feature point (in world frame)
            cv::eigen2cv(w_feature_3d, frame.keypoints_3d_.rowRange(0, 3).col(i));
            // landmark position
            int landmark_idx = frame.keypoint_landmark_[i].first;
            frame.landmarks_[landmark_idx]->point_3d = w_feature_3d;
        }
    }
}


