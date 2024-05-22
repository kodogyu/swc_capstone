#include "visual_odometry.hpp"

VisualOdometry::VisualOdometry(std::string config_path) {
    // parse config file
    pConfig_ = std::make_shared<Configuration>(config_path);
    pConfig_->parse();

    // initialize variables
    pUtils_ = std::make_shared<Utils>(pConfig_);
    pVisualizer_ = std::make_shared<Visualizer>(pConfig_, pUtils_);
    pCamera_ = std::make_shared<Camera>(pConfig_);

    orb_ = cv::ORB::create(pConfig_->num_features_, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 25);
    orb_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    sift_ = cv::SIFT::create();
    sift_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
}

VisualOdometry::~VisualOdometry() {
}

void VisualOdometry::run() {
    std::cout << "VisualOdometry::run" << std::endl;
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
    frame_window_.push_back(pPrev_frame);

    for (int run_iter = 1; run_iter < pConfig_->num_frames_; run_iter++) {
        // start timer [total time cost]
        std::chrono::time_point<std::chrono::steady_clock> total_time_start = std::chrono::steady_clock::now();

        image1_left = cv::imread(pConfig_->left_image_entries_[run_iter], cv::IMREAD_GRAYSCALE);
        // image1_left = readImage(i);
        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        pCurr_frame->image_ = image1_left;
        pCurr_frame->frame_image_idx_ = pConfig_->frame_offset_ + run_iter;
        pCurr_frame->pCamera_ = pCamera_;
        pCurr_frame->pPrevious_frame_ = pPrev_frame;
        pPrev_frame->pNext_frame_ = pCurr_frame;

        //**========== 1. Feature extraction ==========**//
        // start timer [feature extraction]
        Timer feature_timer;
        feature_timer.start();

        cv::Mat curr_image_descriptors;
        std::vector<cv::KeyPoint> curr_image_keypoints;

        if (run_iter == 1) {  // first run
            cv::Mat prev_image_descriptors;
            std::vector<cv::KeyPoint> prev_image_keypoints;
            detectAndCompute(pPrev_frame->image_, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
            pPrev_frame->setKeypointsAndDescriptors(prev_image_keypoints, prev_image_descriptors);
            pUtils_->drawKeypoints(pPrev_frame);
        }
        detectAndCompute(pCurr_frame->image_, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
        pCurr_frame->setKeypointsAndDescriptors(curr_image_keypoints, curr_image_descriptors);
        pUtils_->drawKeypoints(pCurr_frame);

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
        feature_extraction_costs_.push_back(feature_timer.stop());

        //**========== 2. Feature matching ==========**//
        // start timer [feature matching]
        Timer feature_matching_timer;
        feature_matching_timer.start();


        std::vector<cv::DMatch> good_matches;  // good matchings

        int raw_matches_size = -1;
        int good_matches_size = -1;

        // extract points from keypoints
        std::vector<cv::Point2f> image0_kp_pts;
        std::vector<cv::Point2f> image1_kp_pts;

        // image0 & image1 (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches01_vec;
        std::vector<std::vector<cv::DMatch>> image_matches10_vec;
        knnMatch(pPrev_frame->descriptors_, pCurr_frame->descriptors_, image_matches01_vec, 2);  // prev -> curr matches
        knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches10_vec, 2);  // curr -> prev matches

        raw_matches_size = image_matches01_vec.size();

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

        good_matches_size = good_matches.size();

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


        for (auto match : good_matches) {
            image0_kp_pts.push_back(pPrev_frame->keypoints_pt_[match.queryIdx]);
            image1_kp_pts.push_back(pCurr_frame->keypoints_pt_[match.trainIdx]);
        }

        // set frame matches
        pCurr_frame->setFrameMatches(good_matches);

        std::cout << "original features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << raw_matches_size << std::endl;
        std::cout << "good features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << good_matches_size << std::endl;

        // essential matrix
        cv::Mat essential_mat;
        cv::Mat essential_mask;

        essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, pCamera_->intrinsic_, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
        // Eigen::Matrix3d essential_mat_eigen = pUtils_->findEssentialMat(pCamera_->intrinsic_, image0_kp_pts, image1_kp_pts, essential_mask, 0.999, 1.0);

        std::cout << "essential matrix:\n" << essential_mat << std::endl;

        int essential_inliers = pUtils_->countMask(essential_mask);
        std::cout << "essential inliers / matches: " << essential_inliers << " / " << good_matches_size << std::endl;

        // end timer [feature matching]
        feature_matching_costs_.push_back(feature_matching_timer.stop());

        // draw matches
        pUtils_->drawMatches(pPrev_frame, pCurr_frame, good_matches);

        //**========== 3. Motion estimation ==========**//
        // start timer [motion estimation]
        Timer motion_estimation_timer;
        motion_estimation_timer.start();

        Eigen::Isometry3d relative_pose;
        cv::Mat pose_mask = essential_mask.clone();
        cv::Mat R, t;

        pUtils_->recoverPose(pCamera_->intrinsic_, essential_mat, image0_kp_pts, image1_kp_pts, relative_pose, pose_mask);

        std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;

        pCurr_frame->relative_pose_ = relative_pose;
        pCurr_frame->pose_ = pPrev_frame->pose_ * relative_pose;
        poses_.push_back(poses_[run_iter - 1] * relative_pose);
        relative_poses_.push_back(relative_pose);


        // end timer [motion estimation]
        motion_estimation_costs_.push_back(motion_estimation_timer.stop());

        //**========== 4. Triangulation ==========**//
        // start timer [triangulation]
        Timer triangulation_timer;
        triangulation_timer.start();

        std::vector<Eigen::Vector3d> keypoints_3d;
        triangulate3(pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches, pose_mask, keypoints_3d);

        // end timer [triangulation]
        triangulation_costs_.push_back(triangulation_timer.stop());

        // ----- Calculate & Draw reprojection error
        std::vector<Eigen::Vector3d> triangulated_kps, triangulated_kps_cv;

        pUtils_->triangulateKeyPoints(pCurr_frame, image0_kp_pts, image1_kp_pts, triangulated_kps);
        pUtils_->drawReprojectedLandmarks(pCurr_frame, good_matches, pose_mask, triangulated_kps);
        if (pConfig_->calc_reprojection_error_) {
            double reprojection_error = pUtils_->calcReprojectionError(pCurr_frame, good_matches, pose_mask, triangulated_kps);
            std::cout << "reprojection error: " << reprojection_error << std::endl;
        }

        //**========== 5. Scale estimation ==========**//
        // start timer [scaling]
        Timer scaling_timer;
        scaling_timer.start();

        std::vector<int> scale_mask(pCurr_frame->keypoints_pt_.size(), 0);
        // double est_scale_ratio = estimateScale(pPrev_frame, pCurr_frame, scale_mask);
        double est_scale_ratio = estimateScale2(pPrev_frame, pCurr_frame, scale_mask);
        double gt_scale_ratio = getGTScale(pCurr_frame);
        std::cout << "estimated scale: " << est_scale_ratio << ". GT scale: " << gt_scale_ratio << std::endl;
        scales_.push_back(est_scale_ratio);
        gt_scales_.push_back(gt_scale_ratio);
        // applyScale(pCurr_frame, gt_scale_ratio, scale_mask);
        // applyScale(pCurr_frame, est_scale_ratio, scale_mask);

        // end timer [scaling]
        scaling_costs_.push_back(scaling_timer.stop());

        //**========== 6. Local optimization ==========**//
        // start timer [optimization]
        Timer optimization_timer;
        optimization_timer.start();

        if (pConfig_->do_optimize_) {
            frame_window_.push_back(pCurr_frame);

            if (frame_window_.size() > pConfig_->window_size_) {
                frame_window_.erase(frame_window_.begin());
            }
            if (frame_window_.size() == pConfig_->window_size_) {
                double reprojection_error = pUtils_->calcReprojectionError(frame_window_);

                std::cout << "calculated reprojection error: " << reprojection_error << std::endl;

                // Optimize (BA)
                optimizer_.optimizeFrames(frame_window_, pConfig_->optimizer_verbose_);
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
        optimization_costs_.push_back(optimization_timer.stop());

        // logger_.logTrajectoryTxtAppend(pCurr_frame->pose_);
        // logger_.logTimecostAppend(feature_extraction_cost,
        //                     feature_matching_cost,
        //                     motion_estimation_cost,
        //                     triangulation_cost,
        //                     scaling_cost,
        //                     optimization_cost,
        //                     total_time_cost);
    }
    std::cout << "total Landmarks: " << Landmark().total_landmark_cnt_ << std::endl;

    //**========== Log ==========**//
    // trajectory
    // logger_.logTrajectory(relative_poses_);
    logger_.logTrajectory(poses_);
    // logger_.logTrajectory(aligned_est_poses);
    // logger_.logTrajectoryTxt(poses_);
    logger_.logTrajectoryTxt(frames_);

    // time cost[us]
    logger_.logTimecosts(feature_extraction_costs_,
                        feature_matching_costs_,
                        motion_estimation_costs_,
                        triangulation_costs_,
                        scaling_costs_,
                        optimization_costs_,
                        total_time_costs_);

    // scales
    logger_.logScales(scales_, gt_scales_);

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
    if (!(pConfig_->display_type_ & DisplayType::REALTIME_VIS)) {
        std::cout << "frames_ count: " << frames_.size() << std::endl;
        pVisualizer_->setFrameBuffer(frames_);
        pVisualizer_->display(pConfig_->display_type_);
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
    std::cout << "----- VisualOdometry::triangulate3 -----" << std::endl;

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
            cv::Point2f prev_frame_kp_pt = pPrev_frame->keypoints_pt_[good_matches[i].queryIdx];
            cv::Point2f curr_frame_kp_pt = pCurr_frame->keypoints_pt_[good_matches[i].trainIdx];

            // keypoint가 이전 프레임에서 이제까지 한번도 landmark로 선택되지 않았는지 확인 (보존하고 있는 frame에 한해서 검사)
            bool found_landmark = false;
            int kp_idx = good_matches[i].queryIdx;
            std::shared_ptr<Frame> pPrevNFrame = pPrev_frame;
            int landmark_frame_id = -1;
            int landmark_id = -1;

            while (pPrevNFrame->id_ > 0) {
                // std::cout << "[debug] pPrevNFrame id: " << pPrevNFrame->id_ << std::endl;

                landmark_frame_id = pPrevNFrame->id_;
                landmark_id = pPrevNFrame->keypoint_landmark_.at(kp_idx).second;
                found_landmark = (landmark_id != -1);

                // std::cout << "[debug] kp_idx: " << kp_idx << std::endl;
                // std::cout << "[debug] landmark_frame_id: " << landmark_frame_id << std::endl;
                // std::cout << "[debug] landmark_id: " << landmark_id << std::endl;
                // std::cout << "[debug] found_landmark: " << found_landmark << std::endl;

                if (found_landmark) {
                    break;
                }

                kp_idx = pPrevNFrame->matches_with_prev_frame_.at(kp_idx);
                pPrevNFrame = pPrevNFrame->pPrevious_frame_.lock();

                if(kp_idx < 0) {
                    break;
                }
            }

            // hard matching
            if (found_landmark) {  // landmark already exists
                new_landmark = false;

                int landmark_idx = pPrevNFrame->keypoint_landmark_[kp_idx].first;
                std::shared_ptr<Landmark> pLandmark = pPrevNFrame->landmarks_[landmark_idx];

                // compare landmark descriptor similarity for all other frames
                cv::Mat new_descriptors = pCurr_frame->descriptors_;
                int new_desc_idx = good_matches[i].trainIdx;
                // std::cout << "[debug] landmark observation size: " << pLandmark->observations_.size() << std::endl;

                for (std::pair<int, int> observation : pLandmark->observations_) {
                // for (std::map<int, int>::iterator map_itr = pLandmark->observations_.begin(); map_itr != pLandmark->observations_.end(); map_itr++) {
                    int frame_id = observation.first;
                    int target_desc_idx = observation.second;
                    // std::cout << "[debug] frame_id: " << frame_id << std::endl;
                    // std::cout << "[debug] target_desc_idx: " << target_desc_idx << std::endl;

                    cv::Mat target_descriptors = frames_[frame_id]->descriptors_;

                    std::vector<cv::DMatch> match_new_target, match_target_new;
                    match(new_descriptors, target_descriptors, match_new_target);
                    match(target_descriptors, new_descriptors, match_target_new);

                    if ((match_new_target[new_desc_idx].trainIdx != target_desc_idx) ||  // new -> target에서 매칭되어야 함
                            (match_target_new[target_desc_idx].trainIdx != new_desc_idx)) {  // target -> new에서도 매칭되어야 함
                        landmark_accepted = false;
                        break;
                    }
                    // std::cout << "[debug] -----" << std::endl;
                }
            }

            // execute triangulation
            Eigen::Vector3d w_keypoint_3d;  // keypoint coordinate in world frame
            pUtils_->triangulateKeyPoint(pCurr_frame, prev_frame_kp_pt, curr_frame_kp_pt, w_keypoint_3d);

            // std::cout << "[debug] new_landmark: " << new_landmark << std::endl;
            // std::cout << "[debug] landmark_accepted: " << landmark_accepted << std::endl;

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations_.insert({pPrev_frame->id_, good_matches[i].queryIdx});
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
                pLandmark->point_3d_ = w_keypoint_3d;
                // std::cout << "[debug] new_landmark-----" << std::endl;

                pPrev_frame->landmarks_.push_back(pLandmark);
                pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx] = std::pair(pPrev_frame->landmarks_.size() - 1, pLandmark->id_);
                // pPrev_frame->keypoints_3d_[good_matches[i].queryIdx] = w_keypoint_3d;
                pCurr_frame->landmarks_.push_back(pLandmark);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, pLandmark->id_);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                frame_keypoints_3d.push_back(w_keypoint_3d);
                // std::cout << "[debug] -----" << std::endl;
            }
            else if (landmark_accepted) {
                // add information to curr_frame
                std::pair<int, int> prev_n_frame_kp_lm = pPrevNFrame->keypoint_landmark_[kp_idx];
                pCurr_frame->landmarks_.push_back(pPrevNFrame->landmarks_[prev_n_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, landmark_id);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;
                // std::cout << "[debug] landmark_accepted-----" << std::endl;

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrevNFrame->landmarks_[prev_n_frame_kp_lm.first];
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});

                frame_keypoints_3d.push_back(w_keypoint_3d);
                // std::cout << "[debug] -----" << std::endl;
            }
        }
    }
}

double VisualOdometry::calcCovisibleLandmarkDistance(const std::shared_ptr<Frame> &pFrame, const std::vector<int> &covisible_feature_idxs) {
    std::cout << "----- VisualOdometry::calcCovisibleLandmarkDistance -----" << std::endl;
    double acc_distance = 0;

    std::cout << "covisible feature idxs size: " << covisible_feature_idxs.size() << std::endl;
    std::cout << "keypoint_3d size: " << pFrame->keypoints_3d_.size() << std::endl;


    Eigen::Vector3d landmark_3d_k1, landmark_3d_k;
    int landmark_idx = pFrame->keypoint_landmark_[covisible_feature_idxs[0]].first;
    int landmark_id = pFrame->keypoint_landmark_[covisible_feature_idxs[0]].second;


    std::cout << "landmark_idx [0]: " << landmark_idx << std::endl;
    std::cout << "landmark_id [0]: " << landmark_id << std::endl;

    landmark_3d_k1 = pFrame->keypoints_3d_[covisible_feature_idxs[0]];
    std::cout << "  feature index: " << covisible_feature_idxs[0] << std::endl;
    std::cout << "  coordinate: " << landmark_3d_k1.transpose() << std::endl;

    for (int i = 1; i < covisible_feature_idxs.size(); i++) {  // i = landmark index
        landmark_idx = pFrame->keypoint_landmark_[covisible_feature_idxs[i]].first;
        landmark_id = pFrame->keypoint_landmark_[covisible_feature_idxs[i]].second;

        landmark_3d_k = pFrame->keypoints_3d_[covisible_feature_idxs[i]];

        // print landmark id
        std::cout << "landmark_id[" << i << "]: " << landmark_id << std::endl;
        std::cout << "  feature index: " << covisible_feature_idxs[i] << std::endl;
        std::cout << "  coordinate: " << landmark_3d_k.transpose() << std::endl;

        acc_distance += (landmark_3d_k - landmark_3d_k1).norm();

        landmark_3d_k1 = landmark_3d_k;
    }

    std::cout << "acc_distance: " << acc_distance << std::endl;

    return acc_distance;
}

double VisualOdometry::estimateScale(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask) {
    std::cout << "----- VisualOdometry::estimateScale -----" << std::endl;
    double scale_ratio = 1.0;

    std::vector<int> prev_frame_covisible_feature_idxs, curr_frame_covisible_feature_idxs;

    if (pPrev_frame->id_ == 0) {
        return 1.0;
    }

    std::shared_ptr<Frame> pBefore_prev_frame = pPrev_frame->pPrevious_frame_.lock();
    for (auto pLandmark : pPrev_frame->landmarks_) {

        std::cout << "landmark [" << pLandmark->id_ << "] observations: " << std::endl;
        for (auto observation : pLandmark->observations_) {
            std::cout << "  (" << observation.first << ") " << observation.second << std::endl;
        }

        if (pLandmark->observations_.find(pCurr_frame->id_) != pLandmark->observations_.end()  // 현재 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pPrev_frame->id_) != pLandmark->observations_.end()  // 이전 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pBefore_prev_frame->id_) != pLandmark->observations_.end()) {  // 전전 프레임에서 관측되는 landmark
            int curr_frame_feature_idx = pLandmark->observations_.find(pCurr_frame->id_)->second;
            
            prev_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pPrev_frame->id_)->second);
            curr_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pCurr_frame->id_)->second);

            scale_mask[curr_frame_feature_idx] = 1;
        }
    }

    if (prev_frame_covisible_feature_idxs.size() < 2) {
        std::cout << "Number of covisible landmarks should be greater than or equal to 2. Currently " << prev_frame_covisible_feature_idxs.size() << "." << std::endl;
        return 1.0;
    }

    double prev_frame_landmark_distance = calcCovisibleLandmarkDistance(pPrev_frame, prev_frame_covisible_feature_idxs);
    double curr_frame_landmark_distance = calcCovisibleLandmarkDistance(pCurr_frame, curr_frame_covisible_feature_idxs);

    // scale_ratio = curr_frame_landmark_distance / prev_frame_landmark_distance;
    scale_ratio = prev_frame_landmark_distance / curr_frame_landmark_distance;

    return scale_ratio;
}


double VisualOdometry::estimateScale2(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask) {
    std::cout << "----- VisualOdometry::estimateScale2 -----" << std::endl;

    double scale_ratio = 1.0;

    std::vector<int> before_prev_frame_covisible_feature_idxs, prev_frame_covisible_feature_idxs, curr_frame_covisible_feature_idxs;

    if (pPrev_frame->id_ == 0) {
        return 1.0;
    }

    std::shared_ptr<Frame> pBefore_prev_frame = pPrev_frame->pPrevious_frame_.lock();
    for (auto pLandmark : pPrev_frame->landmarks_) {

        std::cout << "landmark [" << pLandmark->id_ << "] observations: " << std::endl;
        for (auto observation : pLandmark->observations_) {
            std::cout << "  (" << observation.first << ") " << observation.second << std::endl;
        }

        if (pLandmark->observations_.find(pCurr_frame->id_) != pLandmark->observations_.end()  // 현재 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pPrev_frame->id_) != pLandmark->observations_.end()  // 이전 프레임에서 관측되는 landmark 이면서
            && pLandmark->observations_.find(pBefore_prev_frame->id_) != pLandmark->observations_.end()) {  // 전전 프레임에서 관측되는 landmark
            int curr_frame_feature_idx = pLandmark->observations_.find(pCurr_frame->id_)->second;

            before_prev_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pBefore_prev_frame->id_)->second);
            prev_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pPrev_frame->id_)->second);
            curr_frame_covisible_feature_idxs.push_back(pLandmark->observations_.find(pCurr_frame->id_)->second);

            scale_mask[curr_frame_feature_idx] = 1;
        }
    }


    std::vector<cv::Point2f> frame0_kp_pts, frame1_kp_pts, frame2_kp_pts;
    for (int i = 0; i < before_prev_frame_covisible_feature_idxs.size(); i++) {
        frame0_kp_pts.push_back(pBefore_prev_frame->keypoints_pt_.at(before_prev_frame_covisible_feature_idxs[i]));
        frame1_kp_pts.push_back(pPrev_frame->keypoints_pt_.at(prev_frame_covisible_feature_idxs[i]));
        frame2_kp_pts.push_back(pCurr_frame->keypoints_pt_.at(curr_frame_covisible_feature_idxs[i]));

        std::cout << "frame0_kp_pts[" << i << "] " << frame0_kp_pts[i] << std::endl;
        std::cout << "frame1_kp_pts[" << i << "] " << frame1_kp_pts[i] << std::endl;
        std::cout << "frame2_kp_pts[" << i << "] " << frame2_kp_pts[i] << std::endl;
    }

    // draw keypoints
    pUtils_->drawKeypoints(pBefore_prev_frame, frame0_kp_pts, "output_logs/intra_frames", "pBefore_frame");
    pUtils_->drawKeypoints(pPrev_frame, frame1_kp_pts, "output_logs/intra_frames", "pPrev_frame");
    pUtils_->drawKeypoints(pCurr_frame, frame2_kp_pts, "output_logs/intra_frames", "pCurr_frame");

    // triangulate
    std::vector<Eigen::Vector3d> keypoints3d_01, keypoints3d_02;
    pUtils_->triangulateKeyPoints(pPrev_frame, frame0_kp_pts, frame1_kp_pts, keypoints3d_01);
    pUtils_->triangulateKeyPoints(pCurr_frame, frame0_kp_pts, frame2_kp_pts, keypoints3d_02);

    Eigen::Vector3d prev_keypoint_3d = keypoints3d_01[0];
    Eigen::Vector3d curr_keypoint_3d;
    double acc_distance_01 = 0;
    for (int i = 1; i < keypoints3d_01.size(); i++) {  // i = landmark index
        curr_keypoint_3d = keypoints3d_01[i];

        acc_distance_01 += (curr_keypoint_3d - prev_keypoint_3d).norm();

        prev_keypoint_3d = curr_keypoint_3d;
    }

    prev_keypoint_3d = keypoints3d_02[0];
    double acc_distance_02 = 0;
    for (int i = 1; i < keypoints3d_02.size(); i++) {  // i = landmark index
        curr_keypoint_3d = keypoints3d_02[i];

        acc_distance_02 += (curr_keypoint_3d - prev_keypoint_3d).norm();

        prev_keypoint_3d = curr_keypoint_3d;
    }

    if (prev_frame_covisible_feature_idxs.size() < 2) {
        std::cout << "Number of covisible landmarks should be greater than or equal to 2. Currently " << prev_frame_covisible_feature_idxs.size() << "." << std::endl;
        return 1.0;
    }

    scale_ratio = acc_distance_02 / acc_distance_01;

    return scale_ratio;
}

void VisualOdometry::applyScale(std::shared_ptr<Frame> &pFrame, const double scale_ratio, const std::vector<int> &scale_mask) {
    for (int i = 0; i < scale_mask.size(); i++) {
        if (scale_mask[i] == 1) {
            // // depth
            // pFrame->depths_[i] *= scale_ratio;

            // relative pose
            pFrame->relative_pose_.translation() *= scale_ratio;    // apply scale

            // pose
            std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();
            Eigen::Isometry3d prev_pose = pPrev_frame->pose_;
            Eigen::Isometry3d scaled_pose = prev_pose * pFrame->relative_pose_;  // apply scale
            pFrame->pose_ = scaled_pose;

            // // feature_3d
            // Eigen::Vector3d w_feature_3d_old = pFrame->keypoints_3d_[i];
            // Eigen::Vector3d feature_versor;
            // feature_versor[0] = (pFrame->keypoints_pt_[i].x - pFrame->pCamera_->cx_) / pFrame->pCamera_->fx_;
            // feature_versor[1] = (pFrame->keypoints_pt_[i].y - pFrame->pCamera_->cy_) / pFrame->pCamera_->fy_;
            // feature_versor[2] = 1.0;
            // Eigen::Vector3d w_feature_3d = pFrame->pose_ * (feature_versor * pFrame->depths_[i]);  // scaled 3d feature point (in world frame)
            // pFrame->keypoints_3d_[i] = w_feature_3d;

            // // landmark position
            // int landmark_idx = pFrame->keypoint_landmark_[i].first;
            // std::shared_ptr<Landmark> pLandmark = pFrame->landmarks_[landmark_idx];

            // if (pLandmark->point_3d_ == w_feature_3d_old) {
            //     pLandmark->point_3d_ = w_feature_3d;
            // }
        }
    }
}

double VisualOdometry::getGTScale(std::shared_ptr<Frame> pFrame) {
    std::cout << "----- VisualOdometry::getGTScale -----" << std::endl;

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

    // std::cout << "pprev pose: " << pprev_pose.matrix() << std::endl;
    // std::cout << "prev pose: " << prev_pose.matrix() << std::endl;
    // std::cout << "curr pose: " << curr_pose.matrix() << std::endl;

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

void VisualOdometry::detectAndCompute(const cv::Mat &image, cv::Mat mask, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    if (pConfig_->feature_extractor_ == 0) {    // ORB
        orb_->detectAndCompute(image, mask, keypoints, descriptors);
    }
    else if (pConfig_->feature_extractor_ == 1) {   // SIFT
        sift_->detectAndCompute(image, mask, keypoints, descriptors);
    }
}

void VisualOdometry::knnMatch(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<std::vector<cv::DMatch>> &image_matches01_vec, int k) {
    if (pConfig_->feature_extractor_ == 0) {    // ORB
        orb_matcher_->knnMatch(queryDescriptors, trainDescriptors, image_matches01_vec, k);  // prev -> curr matches
    }
    else if (pConfig_->feature_extractor_ == 1) {   // SIFT
        sift_matcher_->knnMatch(queryDescriptors, trainDescriptors, image_matches01_vec, k);  // prev -> curr matches
    }
}

void VisualOdometry::match(const cv::Mat &queryDescriptors, const cv::Mat &trainDescriptors, std::vector<cv::DMatch> &matches) {
    if (pConfig_->feature_extractor_ == 0) {    // ORB
        orb_matcher_->match(queryDescriptors, trainDescriptors, matches);
    }
    else if (pConfig_->feature_extractor_ == 1) {    // SIFT
        sift_matcher_->match(queryDescriptors, trainDescriptors, matches);
    }
}


