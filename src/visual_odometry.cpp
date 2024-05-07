#include "visual_odometry.hpp"

VisualOdometry::VisualOdometry(std::string config_path) {
    // parse config file
    pConfig_ = std::make_shared<Configuration>(config_path);
    pConfig_->parse();

    // initialize variables
    pCamera_ = std::make_shared<Camera>(pConfig_);
    pUtils_ = std::make_shared<Utils>(pConfig_);
    pVisualizer_ = std::make_shared<Visualizer>(pConfig_, pUtils_);

    orb_ = cv::ORB::create(pConfig_->num_features_, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 25);
    orb_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    sift_ = cv::SIFT::create();
    sift_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
}

void VisualOdometry::run() {
    //**========== 0. Image load ==========**//
    // read images
    cv::Mat image0_left, image1_left;

    image0_left = cv::imread(pConfig_->left_image_entries_[0], cv::IMREAD_GRAYSCALE);
    // image0_left = readImage(0);

    poses_.push_back(Eigen::Isometry3d::Identity());
    std::shared_ptr<Frame> pPrev_frame = std::make_shared<Frame>();
    pPrev_frame->image_ = image0_left;
    pPrev_frame->frame_image_idx_ = pConfig_->frame_offset_;
    pPrev_frame->pCamera_ = pCamera_;
    frames_.push_back(pPrev_frame);

    for (int i = 1; i < pConfig_->num_frames_; i++) {
        // start timer [total time cost]
        std::chrono::time_point<std::chrono::steady_clock> total_time_start = std::chrono::steady_clock::now();

        image1_left = cv::imread(pConfig_->left_image_entries_[i], cv::IMREAD_GRAYSCALE);
        // image1_left = readImage(i);
        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        pCurr_frame->image_ = image1_left;
        pCurr_frame->frame_image_idx_ = pConfig_->frame_offset_ + i;
        pCurr_frame->pCamera_ = pCamera_;
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
            orb_->detectAndCompute(pPrev_frame->image_, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
            // sift_->detectAndCompute(pPrev_frame->image_, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
            pPrev_frame->setKeypointsAndDescriptors(prev_image_keypoints, prev_image_descriptors);
        }
        orb_->detectAndCompute(pCurr_frame->image_, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
        // sift_->detectAndCompute(pCurr_frame->image_, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
        pCurr_frame->setKeypointsAndDescriptors(curr_image_keypoints, curr_image_descriptors);

        // filter keypoints
        if (pConfig_->filtering_mode_ == FilterMode::KEYPOINT_FILTERING) {
            pUtils_->filterKeypoints(pPrev_frame);

            // draw grid
            pUtils_->drawGrid(pPrev_frame->image_);

            // draw keypoints
            std::string tail;
            tail = "_(" + std::to_string(pConfig_->patch_width_) + ", " + std::to_string(pConfig_->patch_height_) + ", " + std::to_string(pConfig_->kps_per_patch_) + ")";
            pUtils_->drawKeypoints(pPrev_frame, "output_logs/filtered_keypoints", tail);
        }

        // end timer [feature extraction]
        std::chrono::time_point<std::chrono::steady_clock> feature_extraction_end = std::chrono::steady_clock::now();
        // feature extraction cost (us)
        auto feature_extraction_diff = feature_extraction_end - feature_extraction_start;
        auto feature_extraction_cost = std::chrono::duration_cast<std::chrono::milliseconds>(feature_extraction_diff).count();
        feature_extraction_costs_.push_back(feature_extraction_cost);

        //**========== 2. Feature matching ==========**//
        // start timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_start = std::chrono::steady_clock::now();

        // image0 & image1 (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches01_vec;
        std::vector<std::vector<cv::DMatch>> image_matches10_vec;
        orb_matcher_->knnMatch(pPrev_frame->descriptors_, pCurr_frame->descriptors_, image_matches01_vec, 2);  // prev -> curr matches
        orb_matcher_->knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches10_vec, 2);  // curr -> prev matches
        // sift_matcher_->knnMatch(pPrev_frame->descriptors_, pCurr_frame->descriptors_, image_matches01_vec, 2);  // prev -> curr matches
        // sift_matcher_->knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches10_vec, 2);  // curr -> prev matches

        std::vector<cv::DMatch> good_matches;  // good matchings
        // // Mark I
        // for (int i = 0; i < image_matches01_vec.size(); i++) {
        //     if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * pConfig_->des_dist_thresh_) {  // prev -> curr match에서 좋은가?
        //         good_matches.push_back(image_matches01_vec[i][0]);
        //     }
        // }

        // Mark II
        for (int i = 0; i < image_matches01_vec.size(); i++) {
            if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * pConfig_->des_dist_thresh_) {  // prev -> curr match에서 좋은가?
                int image1_keypoint_idx = image_matches01_vec[i][0].trainIdx;
                if (image_matches10_vec[image1_keypoint_idx][0].distance < image_matches10_vec[image1_keypoint_idx][1].distance * pConfig_->des_dist_thresh_) {  // curr -> prev match에서 좋은가?
                    if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image1_keypoint_idx][0].trainIdx)
                        good_matches.push_back(image_matches01_vec[i][0]);
                }
            }
        }

        // filter matches
        if (pConfig_->filtering_mode_ == FilterMode::MATCH_FILTERING) {
            pUtils_->filterMatches(pPrev_frame, good_matches);

            // draw grid
            pUtils_->drawGrid(pPrev_frame->image_);

            // draw keypoints
            std::string tail;
            tail = "_(" + std::to_string(pConfig_->patch_width_) + ", " + std::to_string(pConfig_->patch_height_) + ", " + std::to_string(pConfig_->kps_per_patch_) + ")";
            pUtils_->drawKeypoints(pPrev_frame, "output_logs/filtered_matches", tail);
        }

        std::cout << "original features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << image_matches01_vec.size() << std::endl;
        std::cout << "good features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << good_matches.size() << std::endl;
        // cv::Mat image_matches;
        // cv::drawMatches(image0_left, prev_image_keypoints, image1_left, curr_image_keypoints, good_matches, image_matches);
        // cv::imwrite("output_logs/inter_frames/frame"
        //         + std::to_string(pPrev_frame->frame_image_idx_)
        //         + "&"
        //         + std::to_string(pCurr_frame->frame_image_idx_)
        //         + "_kp_matches(raw).png", image_matches);

        // extract points from keypoints
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

        // essential matrix
        cv::Mat essential_mat;
        cv::Mat essential_mask;

        // TEST
        if (pConfig_->test_mode_) {
            std::cout << "\n[test mode] custom essential matrix finding" << std::endl;
            Eigen::Matrix3d essential_mat_eigen = pUtils_->findEssentialMat(pCamera_->intrinsic_, image0_kp_pts, image1_kp_pts, essential_mask, 0.999, 0.05);
            cv::eigen2cv(essential_mat_eigen, essential_mat);
            // essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, pCamera_->intrinsic_, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
        }
        else {
            essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, pCamera_->intrinsic_, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
            // essential_mat = cv::findEssentialMat(image1_kp_pts, image0_kp_pts, pCamera_->intrinsic_, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
            // cv::Point2d pp(pCamera_->cx_, pCamera_->cy_);
            // essential_mat = cv::findEssentialMat(image1_kp_pts, image0_kp_pts, pCamera_->fx_, pp, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
        }
        std::cout << "essential matrix:\n" << essential_mat << std::endl;

        cv::Mat ransac_matches;
        cv::drawMatches(pPrev_frame->image_, pPrev_frame->keypoints_,
                        pCurr_frame->image_, pCurr_frame->keypoints_,
                        good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), essential_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::putText(ransac_matches, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + " & frame" + std::to_string(pCurr_frame->frame_image_idx_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::imwrite("output_logs/inter_frames/frame"
                + std::to_string(pPrev_frame->frame_image_idx_)
                + "&"
                + std::to_string(pCurr_frame->frame_image_idx_)
                + "_kp_matches(ransac).png", ransac_matches);

        int essential_inliers = pUtils_->countMask(essential_mask);
        std::cout << "essential inliers / matches: " << essential_inliers << " / " << good_matches.size() << std::endl;

        // end timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_end = std::chrono::steady_clock::now();
        // feature matching cost (us)
        auto feature_matching_diff = feature_matching_end - feature_matching_start;
        auto feature_matching_cost = std::chrono::duration_cast<std::chrono::milliseconds>(feature_matching_diff).count();
        feature_matching_costs_.push_back(feature_matching_cost);

        //**========== 3. Motion estimation ==========**//
        // start timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_start = std::chrono::steady_clock::now();

        Eigen::Isometry3d relative_pose;
        cv::Mat pose_mask = essential_mask.clone();

        // TEST
        if (pConfig_->test_mode_) {
            std::cout << "\n[test mode] custom recover pose" << std::endl;
            pUtils_->recoverPose(pCamera_->intrinsic_, essential_mat, image0_kp_pts, image1_kp_pts, relative_pose, pose_mask);
        }
        else {
            cv::Mat R, t;
            cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, pCamera_->intrinsic_, R, t, pose_mask);
            // cv::recoverPose(essential_mat, image1_kp_pts, image0_kp_pts, pCamera_->intrinsic_, R, t, pose_mask);

            Eigen::Matrix3d rotation_mat;
            Eigen::Vector3d translation_mat;

            cv::cv2eigen(R, rotation_mat);
            cv::cv2eigen(t, translation_mat);

            // 필요 없나?
            // gtsam::Rot3 rotation = gtsam::Rot3(rotation_mat);
            // gtsam::Point3 translation = gtsam::Point3(translation_mat);

            relative_pose.linear() = rotation_mat.transpose();
            relative_pose.translation() = - rotation_mat.transpose() * translation_mat;
        }
        std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;

        pCurr_frame->relative_pose_ = relative_pose;
        pCurr_frame->pose_ = pPrev_frame->pose_ * relative_pose;
        poses_.push_back(poses_[i - 1] * relative_pose);
        relative_poses_.push_back(relative_pose);


        // end timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_end = std::chrono::steady_clock::now();
        // motion estimation cost (us)
        auto motion_estimation_diff = motion_estimation_end - motion_estimation_start;
        auto motion_estimation_cost = std::chrono::duration_cast<std::chrono::microseconds>(motion_estimation_diff).count();
        motion_estimation_costs_.push_back(motion_estimation_cost);

        //**========== 4. Triangulation ==========**//
        // start timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_start = std::chrono::steady_clock::now();

        std::vector<Eigen::Vector3d> keypoints_3d;
        // triangulate(camera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches, relative_pose, keypoints_3d);
        // triangulate2(pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches, pose_mask, keypoints_3d);
        triangulate3(pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches, pose_mask, keypoints_3d);
        pPrev_frame->keypoints_3d_ = keypoints_3d;

        // end timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_end = std::chrono::steady_clock::now();
        // motion estimation cost (us)
        auto triangulation_diff = triangulation_end - triangulation_start;
        auto triangulation_cost = std::chrono::duration_cast<std::chrono::milliseconds>(triangulation_diff).count();
        triangulation_costs_.push_back(triangulation_cost);

        std::vector<Eigen::Vector3d> triangulated_kps, triangulated_kps_cv;
        // pUtils_->cv_triangulatePoints(pPrev_frame, image0_kp_pts, pCurr_frame, image1_kp_pts, good_matches, triangulated_kps);
        pUtils_->triangulateKeyPoints(pCurr_frame, image0_kp_pts, image1_kp_pts, triangulated_kps);
        // pUtils_->drawCvReprojectedLandmarks(pPrev_frame, image0_kp_pts, pCurr_frame, image1_kp_pts, triangulated_kps, pose_mask);
        pUtils_->drawReprojectedLandmarks(pCurr_frame, good_matches, pose_mask, triangulated_kps);
        if (pConfig_->calc_reprojection_error_) {
            double reprojection_error = pUtils_->calcReprojectionError(pCurr_frame, good_matches, pose_mask, triangulated_kps);
            std::cout << "reprojection error: " << reprojection_error << std::endl;
        }

        //**========== 5. Scale estimation ==========**//
        // start timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_start = std::chrono::steady_clock::now();

        // std::vector<int> scale_mask(pCurr_frame->keypoints_.size(), 0);
        // double est_scale_ratio = estimateScale(pPrev_frame, pCurr_frame, scale_mask);
        // double gt_scale_ratio = getGTScale(pCurr_frame);
        // std::cout << "estimated scale: " << est_scale_ratio << ". GT scale: " << gt_scale_ratio << std::endl;
        // scales_.push_back(est_scale_ratio);
        // applyScale(pCurr_frame, est_scale_ratio, scale_mask);

        // end timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_end = std::chrono::steady_clock::now();
        // scaling time cost (us)
        auto scaling_diff = scaling_end - scaling_start;
        auto scaling_cost = std::chrono::duration_cast<std::chrono::microseconds>(scaling_diff).count();
        scaling_costs_.push_back(scaling_cost);

        // cv::Mat keypoints_3d_mat = cv::Mat(3, pPrev_frame->keypoints_3d_.size(), CV_64F);
        // for (int i = 0; i < keypoints_3d.size(); i++) {
        //     keypoints_3d_mat.at<double>(0, i) = pPrev_frame->keypoints_3d_[i].x();
        //     keypoints_3d_mat.at<double>(1, i) = pPrev_frame->keypoints_3d_[i].y();
        //     keypoints_3d_mat.at<double>(2, i) = pPrev_frame->keypoints_3d_[i].z();
        // }
        // keypoints_3d_vec_.push_back(keypoints_3d_mat);

        //**========== 6. Local optimization ==========**//
        // start timer [optimization]
        const std::chrono::time_point<std::chrono::steady_clock> optimization_start = std::chrono::steady_clock::now();
        if (pConfig_->do_optimize_) {
            if (pCurr_frame->id_ % 1 == 0) {
                frame_window_.push_back(pCurr_frame);
                if (frame_window_.size() > pConfig_->window_size_) {
                    frame_window_.erase(frame_window_.begin());
                }
                if (frame_window_.size() == pConfig_->window_size_) {
                    double reprojection_error = pUtils_->calcReprojectionError(frame_window_);
                    std::cout << "calculated reprojection error: " << reprojection_error << std::endl;

                    optimizer_.optimizeFrames(frame_window_, pConfig_->optimizer_verbose_);
                }
            }
        }
        // end timer [optimization]
        const std::chrono::time_point<std::chrono::steady_clock> optimization_end = std::chrono::steady_clock::now();
        auto optimization_diff = optimization_end - optimization_start;
        auto optimization_cost = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_diff).count();
        optimization_costs_.push_back(optimization_cost);

        // update visualizer buffer
        if (pConfig_->do_optimize_) {
            pVisualizer_->updateBuffer(frame_window_);
        }
        else {
            pVisualizer_->updateBuffer(pCurr_frame);
        }

        // move on
        frames_.push_back(pCurr_frame);
        pPrev_frame = pCurr_frame;

        // end timer [total time]
        std::chrono::time_point<std::chrono::steady_clock> total_time_end = std::chrono::steady_clock::now();
        // total time cost (ms)
        auto total_time_diff = total_time_end - total_time_start;
        auto total_time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(total_time_diff).count();
        total_time_costs_.push_back(total_time_cost);

        // logger_.logTrajectoryTxtAppend(pCurr_frame->pose_);
        // logger_.logTimecostAppend(feature_extraction_cost,
        //                     feature_matching_cost,
        //                     motion_estimation_cost,
        //                     triangulation_cost,
        //                     scaling_cost,
        //                     optimization_cost,
        //                     total_time_cost);
    }
    // keypoints_3d_vec_.push_back(cv::Mat::zeros(3, 1, CV_64F));
    // pUtils_->drawReprojectedLandmarks(frames_);
    // pUtils_->drawFramesLandmarks(frames_);
    // pUtils_->drawCorrespondingFeatures(frames_, 3, 2);
    std::cout << "total Landmarks: " << Landmark().total_landmark_cnt_ << std::endl;

    //**========== Log ==========**//
    // trajectory
    // logger_.logTrajectory(relative_poses_);
    logger_.logTrajectory(poses_);
    // logger_.logTrajectory(aligned_est_poses);
    // logger_.logTrajectoryTxt(poses_);
    logger_.logTrajectoryTxt(frames_);

    // keypoints
    // logger_.logKeypoints(keypoints_3d_vec_);

    // landmarks
    // logger_.logLandmarks(frames_);

    // time cost[us]
    logger_.logTimecosts(feature_extraction_costs_,
                        feature_matching_costs_,
                        motion_estimation_costs_,
                        triangulation_costs_,
                        scaling_costs_,
                        optimization_costs_,
                        total_time_costs_);

    // scales
    std::vector<double> gt_scales;
    getGTScales(pConfig_->gt_path_, pConfig_->is_kitti_, pConfig_->num_frames_, gt_scales);
    logger_.logScales(scales_, gt_scales);

    // RPE
    std::vector<Eigen::Isometry3d> gt_poses, aligned_est_poses;
    pUtils_->loadGT(gt_poses);
    pUtils_->alignPoses(gt_poses, poses_, aligned_est_poses);

    double rpe_rot, rpe_trans;
    // pUtils_->calcRPE_rt(frames_, rpe_rot, rpe_trans);
    pUtils_->calcRPE_rt(gt_poses, aligned_est_poses, rpe_rot, rpe_trans);
    logger_.logRPE(rpe_rot, rpe_trans);
    std::cout << "RPEr: " << rpe_rot << std::endl;
    std::cout << "RPEt: " << rpe_trans << std::endl;

    //**========== Visualize ==========**//
    switch(pConfig_->display_type_) {
        case DisplayType::POSE_ONLY:
            // pVisualizer_->displayPoses(poses_);
            pVisualizer_->displayPoses(frames_);
            break;
        case DisplayType::POSE_AND_LANDMARKS:
            pVisualizer_->displayFramesAndLandmarks(frames_);
            break;
        case DisplayType::ALIGNED_POSE:
            pVisualizer_->displayPoses(aligned_est_poses);
            break;
    }
}

cv::Mat VisualOdometry::readImage(int img_entry_idx) {
    cv::Mat result;

    // read the image
    cv::Mat image = cv::imread(pConfig_->left_image_entries_[img_entry_idx], cv::IMREAD_GRAYSCALE);

    cv::undistort(image, result, pCamera_->intrinsic_, pCamera_->distortion_);

    // fisheye image processing (rectification)
    if (pConfig_->is_fisheye_) {
        cv::Size new_size(640, 480);
        cv::Mat Knew = (cv::Mat_<double>(3, 3) << new_size.width/4, 0, new_size.width/2,
                                                0, new_size.height/4, new_size.height/2,
                                                0, 0, 1);
        cv::omnidir::undistortImage(image, result, pCamera_->intrinsic_, pCamera_->distortion_, pCamera_->xi_, cv::omnidir::RECTIFY_PERSPECTIVE, Knew, new_size);
    }

    return result;
}

void VisualOdometry::triangulate(cv::Mat camera_Matrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, std::vector<cv::DMatch> good_matches, Eigen::Isometry3d relative_pose, std::vector<Eigen::Vector3d> &frame_keypoints_3d) {

    for (int i = 0; i < good_matches.size(); i++) {  // i = corresponding keypoint index
        cv::Point2f image0_kp_pt = pPrev_frame->keypoints_[good_matches[i].queryIdx].pt;
        cv::Point2f image1_kp_pt = pCurr_frame->keypoints_[good_matches[i].trainIdx].pt;

        Eigen::Vector3d versor;
        versor[0] = (image0_kp_pt.x - camera_Matrix.at<double>(0, 2)) / camera_Matrix.at<double>(0, 0);
        versor[1] = (image0_kp_pt.y - camera_Matrix.at<double>(1, 2)) / camera_Matrix.at<double>(0, 0);
        versor[2] = 1;

        double disparity = image0_kp_pt.x - image1_kp_pt.x;
        if (disparity > 0) {
            bool new_landmark = true;
            // get depth
            double depth = camera_Matrix.at<double>(0, 0) * relative_pose.translation().norm() / disparity;
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

// Mark II
void VisualOdometry::triangulate2(cv::Mat camera_Matrix, std::shared_ptr<Frame> &pPrev_frame, std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> good_matches, const cv::Mat &mask, std::vector<Eigen::Vector3d> &frame_keypoints_3d) {
    Eigen::Matrix3d camera_intrinsic;
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

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

// Mark III
void VisualOdometry::triangulate3(cv::Mat camera_Matrix,
                                    std::shared_ptr<Frame> &pPrev_frame,
                                    std::shared_ptr<Frame> &pCurr_frame,
                                    const std::vector<cv::DMatch> good_matches,
                                    const cv::Mat &mask,
                                    std::vector<Eigen::Vector3d> &frame_keypoints_3d) {
    Eigen::Matrix3d camera_intrinsic;
    Eigen::MatrixXd prev_proj(3, 4);
    Eigen::MatrixXd curr_proj(3, 4);

    cv::cv2eigen(pPrev_frame->pCamera_->intrinsic_, camera_intrinsic);
    prev_proj = camera_intrinsic * pPrev_frame->pose_.matrix().inverse().block<3, 4>(0, 0);
    curr_proj = camera_intrinsic * pCurr_frame->pose_.matrix().inverse().block<3, 4>(0, 0);

    for (int i = 0; i < good_matches.size(); i++) {
        if (mask.at<unsigned char>(i) == 1) {
            bool new_landmark = true;
            bool landmark_accepted = true;
            cv::Point2f prev_frame_kp_pt = pPrev_frame->keypoints_[good_matches[i].queryIdx].pt;
            cv::Point2f curr_frame_kp_pt = pCurr_frame->keypoints_[good_matches[i].trainIdx].pt;

            // hard matching
            if (pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].second != -1) {  // landmark already exists
                new_landmark = false;

                int landmark_idx = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].first;
                auto pLandmark = pPrev_frame->landmarks_[landmark_idx];

                // compare landmark descriptor similarity for all other frames
                cv::Mat new_descriptors = pCurr_frame->descriptors_;
                int new_desc_idx = good_matches[i].trainIdx;
                for (auto observation : pLandmark->observations_) {
                    int frame_id = observation.first;
                    int target_desc_idx = observation.second;

                    cv::Mat target_descriptors = frames_[frame_id]->descriptors_;

                    std::vector<cv::DMatch> match_new_target, match_target_new;
                    orb_matcher_->match(new_descriptors, target_descriptors, match_new_target);
                    orb_matcher_->match(target_descriptors, new_descriptors, match_target_new);
                    // sift_matcher_->match(new_descriptors, target_descriptors, match_new_target);
                    // sift_matcher_->match(target_descriptors, new_descriptors, match_target_new);
                    if ((match_new_target[new_desc_idx].trainIdx != target_desc_idx) ||  // new -> target에서 매칭되어야 함
                            (match_target_new[target_desc_idx].trainIdx != new_desc_idx)) {  // target -> new에서도 매칭되어야 함
                        landmark_accepted = false;
                        break;
                    }
                }
            }

            // execute triangulation
            Eigen::Vector3d point_3d;
            pUtils_->triangulateKeyPoint(pCurr_frame, prev_frame_kp_pt, curr_frame_kp_pt, point_3d);
            frame_keypoints_3d.push_back(point_3d);

            Eigen::Vector3d w_keypoint_3d = point_3d;  // keypoint coordinate in world frame

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations_.insert({pPrev_frame->id_, good_matches[i].queryIdx});
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
                pLandmark->point_3d_ = w_keypoint_3d;

                pPrev_frame->landmarks_.push_back(pLandmark);
                pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx] = std::pair(pPrev_frame->landmarks_.size() - 1, pLandmark->id_);
                pPrev_frame->keypoints_3d_[good_matches[i].queryIdx] = w_keypoint_3d;
                pCurr_frame->landmarks_.push_back(pLandmark);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, pLandmark->id_);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;
            }
            else if (landmark_accepted) {
                // add information to curr_frame
                std::pair<int, int> prev_frame_kp_lm = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx];
                int landmark_id = prev_frame_kp_lm.second;
                pCurr_frame->landmarks_.push_back(pPrev_frame->landmarks_[prev_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size(), landmark_id);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrev_frame->landmarks_[prev_frame_kp_lm.first];
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
            }
        }
    }
}

double VisualOdometry::calcCovisibleLandmarkDistance(const Frame &frame, const std::vector<int> &covisible_feature_idxs) {
    std::cout << "----- VisualOdometry::calcCovisibleLandmarkDistance -----" << std::endl;
    double acc_distance = 0;

    Eigen::Vector3d landmark_3d_k1, landmark_3d_k;
    std::shared_ptr<Frame> pPrev_frame = frame.pPrevious_frame_.lock();
    int landmark_id;

    landmark_3d_k1 = pPrev_frame->keypoints_3d_[frame.landmarks_[frame.keypoint_landmark_[covisible_feature_idxs[0]].first]->observations_.find(pPrev_frame->id_)->second];
    landmark_id = frame.keypoint_landmark_[covisible_feature_idxs[0]].second;
    std::cout << "landmark_id[0]: " << landmark_id << std::endl;
    for (int i = 1; i < covisible_feature_idxs.size(); i++) {  // i = landmark index
        int curr_frame_landmark_idx = frame.keypoint_landmark_[covisible_feature_idxs[i]].first;
        int prev_frame_keypoint_idx = frame.landmarks_[curr_frame_landmark_idx]->observations_.find(pPrev_frame->id_)->second;

        landmark_3d_k = pPrev_frame->keypoints_3d_[prev_frame_keypoint_idx];

        // print landmark id
        landmark_id = frame.keypoint_landmark_[covisible_feature_idxs[i]].second;
        std::cout << "landmark_id[" << i << "]: " << landmark_id << std::endl;

        acc_distance += (landmark_3d_k - landmark_3d_k1).norm();

        landmark_3d_k1 = landmark_3d_k;
    }

    return acc_distance;
}

double VisualOdometry::estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask) {
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

void VisualOdometry::applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask) {
    for (int i = 0; i < scale_mask.size(); i++) {
        if (scale_mask[i] == 1) {
            // depth
            pFrame->depths_[i] *= scale_ratio;

            // relative pose
            Eigen::Isometry3d relative_pose_old = pFrame->relative_pose_;  // copy relative pose
            pFrame->relative_pose_.translation() *= scale_ratio;    // apply scale

            // pose
            std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();
            Eigen::Isometry3d prev_pose = pPrev_frame->pose_;
            Eigen::Isometry3d scaled_pose = prev_pose * pFrame->relative_pose_;  // apply scale
            pFrame->pose_ = scaled_pose;

            // feature_3d
            Eigen::Vector3d w_feature_3d_old = pFrame->keypoints_3d_[i];
            Eigen::Vector3d feature_versor;
            feature_versor[0] = (pFrame->keypoints_[i].pt.x - pFrame->pCamera_->cx_) / pFrame->pCamera_->fx_;
            feature_versor[1] = (pFrame->keypoints_[i].pt.y - pFrame->pCamera_->cy_) / pFrame->pCamera_->fy_;
            feature_versor[2] = 1.0;
            Eigen::Vector3d w_feature_3d = pFrame->pose_ * (feature_versor * pFrame->depths_[i]);  // scaled 3d feature point (in world frame)
            pFrame->keypoints_3d_[i] = w_feature_3d;

            // landmark position
            int landmark_idx = pFrame->keypoint_landmark_[i].first;
            std::shared_ptr<Landmark> pLandmark = pFrame->landmarks_[landmark_idx];

            if (pLandmark->point_3d_ == w_feature_3d_old) {
                pLandmark->point_3d_ = w_feature_3d;
            }
        }
    }
}

double VisualOdometry::getGTScale(std::shared_ptr<Frame> pFrame) {
    std::cout << "GT scale function" << std::endl;

    if (pFrame->id_ == 1) {
        return 1.0;
    }
    int curr_id = pFrame->id_;
    int prev_id = curr_id - 1;
    int pprev_id = curr_id - 2;

    Eigen::Isometry3d pprev_pose, prev_pose, curr_pose;

    std::cout << "reading file" << std::endl;
    pprev_pose = pUtils_->getGT(pprev_id);
    prev_pose = pUtils_->getGT(prev_id);
    curr_pose = pUtils_->getGT(curr_id);

    std::cout << "pprev pose: " << pprev_pose.matrix() << std::endl;
    std::cout << "prev pose: " << prev_pose.matrix() << std::endl;
    std::cout << "curr pose: " << curr_pose.matrix() << std::endl;

    // double gt_scale = (curr_pose.translation() - prev_pose.translation()).norm() / (prev_pose.translation() - pprev_pose.translation()).norm();
    double gt_scale = (prev_pose.translation() - pprev_pose.translation()).norm() / (curr_pose.translation() - prev_pose.translation()).norm();
    return gt_scale;
}

void VisualOdometry::getGTScales(const std::string gt_path, bool is_kitti, int num_frames, std::vector<double> &gt_scales) {
    std::ifstream gt_poses__file(gt_path);
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    Eigen::Vector3d prev_position, curr_position;
    double prev_trans_length;
    double trans_length;
    double scale;

    for (int i = 0; i < num_frames; i++) {
        std::getline(gt_poses__file, line);
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


