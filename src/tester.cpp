#include "tester.hpp"
#include "visual_odometry.hpp"

Tester::Tester() {
    // int
    // manual_kp_frame2_ = {cv::Point2f(65, 90), cv::Point2f(270, 149), cv::Point2f(434, 208), cv::Point2f(436, 163), cv::Point2f(498, 191), cv::Point2f(668, 151), cv::Point2f(720, 136), cv::Point2f(827, 125), cv::Point2f(939, 108), cv::Point2f(977, 324)};
    // manual_kp_frame3_ = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106), cv::Point2f(1035, 346)};
    // manual_kp_frame4_ = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104),    cv::Point2f(653, 124)};

    // float
    // original 10 points
    // manual_kp_frame0_ = {cv::Point2f(89.0, 92.0), cv::Point2f(279.65207, 147.67671), cv::Point2f(440.3918, 203.43727), cv::Point2f(434.0, 161.0), cv::Point2f(486.80063, 190.52914), cv::Point2f(661.3766, 148.82133), cv::Point2f(709.0, 136.0), cv::Point2f(807.58307, 125.778046), cv::Point2f(916.523, 110.59789), cv::Point2f(907.6637, 294.75223)};
    // manual_kp_frame1_ = {cv::Point2f(76.435524, 92.402176), cv::Point2f(274.63477, 149.50064), cv::Point2f(439.3174, 206.34424), cv::Point2f(435.0, 163.0), cv::Point2f(487.71252, 192.5296), cv::Point2f(664.4662, 149.73691), cv::Point2f(714.8803, 135.63571), cv::Point2f(816.14087, 125.12716), cv::Point2f(928.7381, 109.08112), cv::Point2f(942.7255, 307.60168)};
    // manual_kp_frame2_ = {cv::Point2f(64.17699, 89.969376), cv::Point2f(270.2492, 149.04694), cv::Point2f(439.1965, 207.44862), cv::Point2f(436.0, 163.0), cv::Point2f(490.15625, 193.59467), cv::Point2f(668.5607, 150.48264), cv::Point2f(720.21185, 135.68932), cv::Point2f(826.4441, 124.59228), cv::Point2f(942.6891, 107.84003), cv::Point2f(987.5414, 324.60016)};
    // manual_kp_frame3_ = {cv::Point2f(50.730343, 88.511986), cv::Point2f(265.05374, 149.24902), cv::Point2f(438.2673, 209.64577), cv::Point2f(437.0, 164.0), cv::Point2f(491.30927, 194.9783), cv::Point2f(672.4143, 151.02629), cv::Point2f(725.9593, 135.52324), cv::Point2f(837.37067, 123.57032), cv::Point2f(958.5006, 106.30311), cv::Point2f(1046.6061, 345.912)};
    // manual_kp_frame4_ = {cv::Point2f(36.555363, 84.93439), cv::Point2f(260.22626, 148.3548), cv::Point2f(437.4876, 210.6658), cv::Point2f(439.0, 164.0), cv::Point2f(494.2114, 195.0669), cv::Point2f(679.5286, 150.93016), cv::Point2f(732.82166, 134.72882), cv::Point2f(850.3517, 122.02151), cv::Point2f(975.88513, 104.29728),    cv::Point2f(652.8191, 123.62265)};

    // 골고루 분포
    // 10개
    manual_kp_frame0_ = {cv::Point2f(123.885864, 141.05359), cv::Point2f(226.58939, 123.59321), cv::Point2f(391.45505, 300.2994), cv::Point2f(493.1157, 278.21002), cv::Point2f(550.3434, 267.6215), cv::Point2f(445.11765, 160.5985), cv::Point2f(661.3764, 148.8213), cv::Point2f(685.77405, 188.34718), cv::Point2f(771.3757, 157.38417), cv::Point2f(916.52295, 110.59792)};
    manual_kp_frame1_ = {cv::Point2f(108.14026, 142.31174), cv::Point2f(214.90706, 124.48146), cv::Point2f(379.383, 310.503), cv::Point2f(488.2138, 285.5208), cv::Point2f(549.38214, 274.37592), cv::Point2f(446.3624, 162.47493), cv::Point2f(664.46606, 149.7369), cv::Point2f(690.72107, 190.05965), cv::Point2f(778.8773, 157.77487), cv::Point2f(928.7381, 109.08129)};
    manual_kp_frame2_ = {cv::Point2f(88.81013, 139.45332), cv::Point2f(202.65263, 123.19561), cv::Point2f(365.405, 321.05316), cv::Point2f(484.0056, 293.48444), cv::Point2f(549.3535, 280.53455), cv::Point2f(447.76715, 163.21419), cv::Point2f(668.55725, 150.48215), cv::Point2f(696.5336, 191.44913), cv::Point2f(787.9124, 158.30823), cv::Point2f(942.6891, 107.84003)};
    // 20개
    manual_kp_frame0_ = {cv::Point2f(123.885864, 141.05359), cv::Point2f(226.58939, 123.59321), cv::Point2f(391.45505, 300.2994), cv::Point2f(249.40048, 220.55078), cv::Point2f(395.2872, 245.68948), cv::Point2f(493.1157, 278.21002), cv::Point2f(550.3434, 267.6215), cv::Point2f(445.11765, 160.5985), cv::Point2f(375.5274, 144.35362), cv::Point2f(478.5863, 111.77609), cv::Point2f(661.3764, 148.8213), cv::Point2f(645.3754, 227.77051), cv::Point2f(629.2117, 270.36212), cv::Point2f(639.9698, 308.2961), cv::Point2f(760.8706, 220.68791), cv::Point2f(771.3757, 157.38417), cv::Point2f(772.40204, 101.60126), cv::Point2f(807.58307, 125.77807), cv::Point2f(869.11084, 143.07722), cv::Point2f(916.52295, 110.59792)};
    manual_kp_frame1_ = {cv::Point2f(108.14026, 142.31174), cv::Point2f(214.90706, 124.48146), cv::Point2f(379.383, 310.503), cv::Point2f(243.09521, 223.60213), cv::Point2f(390.08966, 250.58029), cv::Point2f(488.2138, 285.5208), cv::Point2f(549.38214, 274.37592), cv::Point2f(446.3624, 162.47493), cv::Point2f(374.37546, 150.1095), cv::Point2f(481.19946, 113.4859), cv::Point2f(664.46606, 149.7369), cv::Point2f(648.8862, 230.60934), cv::Point2f(634.0395, 275.2988), cv::Point2f(645.5138, 320.17755), cv::Point2f(770.83026, 224.55469), cv::Point2f(778.8773, 157.77487), cv::Point2f(781.8492, 99.58071), cv::Point2f(816.1409, 125.127235), cv::Point2f(879.67456, 142.6531), cv::Point2f(928.7381, 109.08129)};
    manual_kp_frame2_ = {cv::Point2f(88.81013, 139.45332), cv::Point2f(202.65263, 123.19561), cv::Point2f(365.405, 321.05316), cv::Point2f(237.10286, 224.56015), cv::Point2f(385.02838, 254.22237), cv::Point2f(484.0056, 293.48444), cv::Point2f(549.3535, 280.53455), cv::Point2f(447.76715, 163.21419), cv::Point2f(371.67062, 152.24872), cv::Point2f(484.576, 113.641975), cv::Point2f(668.5607, 150.48264), cv::Point2f(653.2298, 233.36496), cv::Point2f(638.3341, 282.45636), cv::Point2f(652.95123, 333.67984), cv::Point2f(782.7799, 228.21115), cv::Point2f(787.9124, 158.30823), cv::Point2f(794.01935, 96.74404), cv::Point2f(826.4441, 124.59228), cv::Point2f(892.17474, 142.8682), cv::Point2f(942.6891, 107.84003)};
    manual_kps_vec_ = {manual_kp_frame0_, manual_kp_frame1_, manual_kp_frame2_, manual_kp_frame3_, manual_kp_frame4_};

    // 20개
    manual_match_0_1_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9), TestMatch(10, 10), TestMatch(11, 11), TestMatch(12, 12), TestMatch(13, 13), TestMatch(14, 14), TestMatch(15, 15), TestMatch(16, 16), TestMatch(17, 17), TestMatch(18, 18), TestMatch(19, 19)};
    manual_match_1_2_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9), TestMatch(10, 10), TestMatch(11, 11), TestMatch(12, 12), TestMatch(13, 13), TestMatch(14, 14), TestMatch(15, 15), TestMatch(16, 16), TestMatch(17, 17), TestMatch(18, 18), TestMatch(19, 19)};
    manual_match_2_3_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9), TestMatch(10, 10), TestMatch(11, 11), TestMatch(12, 12), TestMatch(13, 13), TestMatch(14, 14), TestMatch(15, 15), TestMatch(16, 16), TestMatch(17, 17), TestMatch(18, 18), TestMatch(19, 19)};
    // 10개
    // manual_match_0_1_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    // manual_match_1_2_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    // manual_match_2_3_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    manual_match_3_4_ = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8)};
    manual_matches_vec_ = {manual_match_0_1_, manual_match_1_2_, manual_match_2_3_, manual_match_3_4_};
}


void Tester::run(VisualOdometry &vo) {
    std::cout << "Tester::run" << std::endl;
    //**========== 0. Image load ==========**//
    // read images
    cv::Mat image0_left, image1_left;

    image0_left = cv::imread(vo.pConfig_->left_image_entries_[0], cv::IMREAD_GRAYSCALE);
    // image0_left = readImage(0);

    vo.poses_.push_back(Eigen::Isometry3d::Identity());
    std::shared_ptr<Frame> pPrev_frame = std::make_shared<Frame>();
    pPrev_frame->image_ = image0_left;
    pPrev_frame->frame_image_idx_ = vo.pConfig_->frame_offset_;
    pPrev_frame->pCamera_ = vo.pCamera_;
    vo.frames_.push_back(pPrev_frame);
    vo.frame_window_.push_back(pPrev_frame);

    for (int run_iter = 1; run_iter < vo.pConfig_->num_frames_; run_iter++) {
        // start timer [total time cost]
        std::chrono::time_point<std::chrono::steady_clock> total_time_start = std::chrono::steady_clock::now();

        image1_left = cv::imread(vo.pConfig_->left_image_entries_[run_iter], cv::IMREAD_GRAYSCALE);
        // image1_left = readImage(i);
        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        pCurr_frame->image_ = image1_left;
        pCurr_frame->frame_image_idx_ = vo.pConfig_->frame_offset_ + run_iter;
        pCurr_frame->pCamera_ = vo.pCamera_;
        pCurr_frame->pPrevious_frame_ = pPrev_frame;
        pPrev_frame->pNext_frame_ = pCurr_frame;

        //**========== 1. Feature extraction ==========**//
        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature extraction" << std::endl;

            if (run_iter == 1) {  // first run
                setFrameKeypoints_pt(pPrev_frame, manual_kp_frame0_);
                vo.pUtils_->drawKeypoints(pPrev_frame);
            }
            setFrameKeypoints_pt(pCurr_frame, manual_kps_vec_[run_iter]);
            vo.pUtils_->drawKeypoints(pCurr_frame);
        }
        //**========== 2. Feature matching ==========**//
        // start timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_start = std::chrono::steady_clock::now();

        std::vector<TestMatch> good_matches_test;  // for TEST mode
        std::vector<cv::DMatch> good_matches;  // good matchings

        int raw_matches_size = -1;
        int good_matches_size = -1;

        // extract points from keypoints
        std::vector<cv::Point2f> image0_kp_pts;
        std::vector<cv::Point2f> image1_kp_pts;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature matching" << std::endl;

            good_matches_test = manual_matches_vec_[run_iter - 1];

            for (auto match : good_matches_test) {
                image0_kp_pts.push_back(pPrev_frame->keypoints_pt_[match.queryIdx]);
                image1_kp_pts.push_back(pCurr_frame->keypoints_pt_[match.trainIdx]);
            }

            // set keyframe matches
            setFrameMatches(pCurr_frame, good_matches_test);

            raw_matches_size = manual_matches_vec_[run_iter - 1].size();
            good_matches_size = good_matches_test.size();
        }
        // set frame matches
        pCurr_frame->setFrameMatches(good_matches);

        std::cout << "original features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << raw_matches_size << std::endl;
        std::cout << "good features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << good_matches_size << std::endl;

        // essential matrix
        cv::Mat essential_mat;
        cv::Mat essential_mask;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            // std::cout << "\n[test mode] custom essential matrix finding" << std::endl;

            // Eigen::Matrix3d essential_mat_eigen = pUtils_->findEssentialMat(pCamera_->intrinsic_, image0_kp_pts, image1_kp_pts, essential_mask, 0.999, 0.05);
            // cv::eigen2cv(essential_mat_eigen, essential_mat);

            essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, vo.pCamera_->intrinsic_, cv::RANSAC, 0.999, 1.0, 500, essential_mask);
        }
        std::cout << "essential matrix:\n" << essential_mat << std::endl;

        int essential_inliers = vo.pUtils_->countMask(essential_mask);
        std::cout << "essential inliers / matches: " << essential_inliers << " / " << good_matches_size << std::endl;

        // end timer [feature matching]
        std::chrono::time_point<std::chrono::steady_clock> feature_matching_end = std::chrono::steady_clock::now();
        auto feature_matching_diff = feature_matching_end - feature_matching_start;
        auto feature_matching_cost = std::chrono::duration_cast<std::chrono::milliseconds>(feature_matching_diff).count();  // feature matching cost (ms)
        vo.feature_matching_costs_.push_back(feature_matching_cost);

        //**========== 3. Motion estimation ==========**//
        // start timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_start = std::chrono::steady_clock::now();

        Eigen::Isometry3d relative_pose;
        cv::Mat pose_mask = essential_mask.clone();

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Motion estimation" << std::endl;
            vo.pUtils_->recoverPose(vo.pCamera_->intrinsic_, essential_mat, image0_kp_pts, image1_kp_pts, relative_pose, pose_mask);
            // relative_pose = pUtils_->getGT(pPrev_frame->frame_image_idx_).inverse() * pUtils_->getGT(pCurr_frame->frame_image_idx_);
            // if (run_iter == 2) {
            //     // relative_pose.translation() *= 2;
            // }
        }
        std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;

        pCurr_frame->relative_pose_ = relative_pose;
        pCurr_frame->pose_ = pPrev_frame->pose_ * relative_pose;
        vo.poses_.push_back(vo.poses_[run_iter - 1] * relative_pose);
        vo.relative_poses_.push_back(relative_pose);


        // end timer [motion estimation]
        std::chrono::time_point<std::chrono::steady_clock> motion_estimation_end = std::chrono::steady_clock::now();
        auto motion_estimation_diff = motion_estimation_end - motion_estimation_start;
        auto motion_estimation_cost = std::chrono::duration_cast<std::chrono::microseconds>(motion_estimation_diff).count();  // motion estimation cost (ms)
        vo.motion_estimation_costs_.push_back(motion_estimation_cost);

        //**========== 4. Triangulation ==========**//
        // start timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_start = std::chrono::steady_clock::now();

        std::vector<Eigen::Vector3d> keypoints_3d;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Triangulation" << std::endl;

            triangulate3(vo, vo.pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches_test, pose_mask, keypoints_3d);
            std::cout << "[debug] Triangulation done." << std::endl;
        }
        // end timer [triangulation]
        std::chrono::time_point<std::chrono::steady_clock> triangulation_end = std::chrono::steady_clock::now();
        auto triangulation_diff = triangulation_end - triangulation_start;
        auto triangulation_cost = std::chrono::duration_cast<std::chrono::milliseconds>(triangulation_diff).count();  // triangulation cost (ms)
        vo.triangulation_costs_.push_back(triangulation_cost);

        // ----- Calculate & Draw reprojection error
        std::vector<Eigen::Vector3d> triangulated_kps, triangulated_kps_cv;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            vo.pUtils_->triangulateKeyPoints(pCurr_frame, image0_kp_pts, image1_kp_pts, triangulated_kps);
            drawReprojectedLandmarks(pCurr_frame, good_matches_test, pose_mask, triangulated_kps);
            if (vo.pConfig_->calc_reprojection_error_) {
                double reprojection_error = calcReprojectionError(pCurr_frame, good_matches_test, pose_mask, triangulated_kps);
                std::cout << "reprojection error: " << reprojection_error << std::endl;
            }
        }
        //**========== 5. Scale estimation ==========**//
        // start timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_start = std::chrono::steady_clock::now();

        std::vector<int> scale_mask(pCurr_frame->keypoints_pt_.size(), 0);
        double est_scale_ratio = vo.estimateScale(pPrev_frame, pCurr_frame, scale_mask);
        double gt_scale_ratio = vo.getGTScale(pCurr_frame);
        std::cout << "estimated scale: " << est_scale_ratio << ". GT scale: " << gt_scale_ratio << std::endl;
        vo.scales_.push_back(est_scale_ratio);
        vo.gt_scales_.push_back(gt_scale_ratio);
        // applyScale(pCurr_frame, gt_scale_ratio, scale_mask);
        // applyScale(pCurr_frame, est_scale_ratio, scale_mask);

        // end timer [scaling]
        std::chrono::time_point<std::chrono::steady_clock> scaling_end = std::chrono::steady_clock::now();
        auto scaling_diff = scaling_end - scaling_start;
        auto scaling_cost = std::chrono::duration_cast<std::chrono::microseconds>(scaling_diff).count();  // scaling time cost (ms)
        vo.scaling_costs_.push_back(scaling_cost);

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
        if (vo.pConfig_->do_optimize_) {
            vo.frame_window_.push_back(pCurr_frame);

            if (vo.frame_window_.size() > vo.pConfig_->window_size_) {
                vo.frame_window_.erase(vo.frame_window_.begin());
            }
            if (vo.frame_window_.size() == vo.pConfig_->window_size_) {
                double reprojection_error = vo.pUtils_->calcReprojectionError(vo.frame_window_);

                std::cout << "calculated reprojection error: " << reprojection_error << std::endl;

                // Optimize (BA)
                vo.optimizer_.optimizeFrames(vo.frame_window_, vo.pConfig_->optimizer_verbose_);
            }
        }
        // end timer [optimization]
        const std::chrono::time_point<std::chrono::steady_clock> optimization_end = std::chrono::steady_clock::now();
        auto optimization_diff = optimization_end - optimization_start;
        auto optimization_cost = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_diff).count();
        vo.optimization_costs_.push_back(optimization_cost);

        // update visualizer buffer
        if (vo.pConfig_->do_optimize_) {
            vo.pVisualizer_->updateBuffer(vo.frame_window_);
        }
        else {
            vo.pVisualizer_->updateBuffer(pCurr_frame);
        }

        // move on
        vo.frames_.push_back(pCurr_frame);
        pPrev_frame = pCurr_frame;

        // end timer [total time]
        std::chrono::time_point<std::chrono::steady_clock> total_time_end = std::chrono::steady_clock::now();
        auto total_time_diff = total_time_end - total_time_start;
        auto total_time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(total_time_diff).count();  // total time cost (ms)
        vo.total_time_costs_.push_back(total_time_cost);

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
    vo.logger_.logTrajectory(vo.poses_);
    // logger_.logTrajectory(aligned_est_poses);
    // logger_.logTrajectoryTxt(poses_);
    vo.logger_.logTrajectoryTxt(vo.frames_);

    // keypoints
    // logger_.logKeypoints(keypoints_3d_vec_);

    // landmarks
    // logger_.logLandmarks(frames_);

    // time cost[us]
    vo.logger_.logTimecosts(vo.feature_extraction_costs_,
                        vo.feature_matching_costs_,
                        vo.motion_estimation_costs_,
                        vo.triangulation_costs_,
                        vo.scaling_costs_,
                        vo.optimization_costs_,
                        vo.total_time_costs_);

    // scales
    // std::vector<double> gt_scales;
    // getGTScales(pConfig_->gt_path_, pConfig_->is_kitti_, pConfig_->num_frames_, gt_scales);
    vo.logger_.logScales(vo.scales_, vo.gt_scales_);

    // RPE
    std::vector<Eigen::Isometry3d> gt_poses, aligned_est_poses;
    vo.pUtils_->loadGT(gt_poses);
    vo.pUtils_->alignPoses(gt_poses, vo.poses_, aligned_est_poses);

    double rpe_rot, rpe_trans;
    // pUtils_->calcRPE_rt(frames_, rpe_rot, rpe_trans);
    vo.pUtils_->calcRPE_rt(gt_poses, aligned_est_poses, rpe_rot, rpe_trans);
    vo.logger_.logRPE(rpe_rot, rpe_trans);
    std::cout << "RPEr: " << rpe_rot << std::endl;
    std::cout << "RPEt: " << rpe_trans << std::endl;

    //**========== Visualize ==========**//
    if (!(vo.pConfig_->display_type_ & DisplayType::REALTIME_VIS)) {
        std::cout << "frames_ count: " << vo.frames_.size() << std::endl;
        vo.pVisualizer_->setFrameBuffer(vo.frames_);
        vo.pVisualizer_->display(vo.pConfig_->display_type_);
    }
}


void Tester::decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat intrinsic, std::vector<cv::Point2f> image0_kp_pts, std::vector<cv::Point2f> image1_kp_pts, const cv::Mat &mask, cv::Mat &_R, cv::Mat &_t) {
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(essential_mat, R1, R2, t);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    std::vector<Eigen::Isometry3d> rel_poses(4, Eigen::Isometry3d::Identity());
    std::vector<int> positive_cnts(4, 0);
    for (int i = 0; i < 4; i++) {
        if (i == 0) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (i == 1) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        else if (i == 2) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (i == 3) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        rel_poses[i].linear() = rotation_mat;
        rel_poses[i].translation() = translation_mat;
        positive_cnts[i] = getPositiveLandmarksCount(intrinsic, image0_kp_pts, image1_kp_pts, rel_poses[i], mask);
        std::cout << "cnt[" << i << "]: " << positive_cnts[i] << std::endl;
        // poses.push_back(rel_poses[i]);
    }

    int max_cnt = 0, max_idx = 0;
    for (int i = 0; i < 4; i++) {
        // std::cout << "cnt[" << i << "]: " << positive_cnts[i] << std::endl;
        if (positive_cnts[i] > max_cnt) {
            max_cnt = positive_cnts[i];
            max_idx = i;
        }
    }
    std::cout << "max idx: " << max_idx << std::endl;

    Eigen::Matrix3d rotation = rel_poses[max_idx].rotation();
    Eigen::Vector3d translation = rel_poses[max_idx].translation();
    cv::eigen2cv(rotation, _R);
    cv::eigen2cv(translation, _t);
}

int Tester::getPositiveLandmarksCount(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask/*, std::vector<Eigen::Vector3d> &landmarks*/) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    int positive_cnt = 0;
    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector3d cam1_point_3d = cam1_pose.inverse().matrix().block<3, 4>(0, 0) * point_3d_homo;
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (cam1_point_3d.z() > 0 && point_3d.z() > 0) {
            positive_cnt++;
        }
    }
    return positive_cnt;
}

void Tester::setFrameKeypoints_pt(const std::shared_ptr<Frame> &pFrame, std::vector<cv::Point2f> kp_pts) {
    std::cout << "----- Tester::setFrameKeypoints_pt -----" << std::endl;

    // init keypoints_pt_
    pFrame->keypoints_pt_ = kp_pts;

    // init 3D keypoints
    pFrame->keypoints_3d_.assign(kp_pts.size(), Eigen::Vector3d(0.0, 0.0, 0.0));

    // init matches
    pFrame->matches_with_prev_frame_ = std::vector<int>(kp_pts.size(), -1);
    pFrame->matches_with_next_frame_ = std::vector<int>(kp_pts.size(), -1);

    // init depths_
    pFrame->depths_.reserve(kp_pts.size());

    // init keypoint_landmark_
    pFrame->keypoint_landmark_.assign(kp_pts.size(), std::pair(-1, -1));
}

void Tester::setFrameMatches(const std::shared_ptr<Frame> &pFrame, const std::vector<TestMatch> &matches_with_prev_frame) {
    int queryIdx, trainIdx;
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    for (int i = 0; i < matches_with_prev_frame.size(); i++) {
        queryIdx = matches_with_prev_frame[i].queryIdx;
        trainIdx = matches_with_prev_frame[i].trainIdx;

        pFrame->matches_with_prev_frame_[queryIdx] = trainIdx;
        pPrev_frame->matches_with_next_frame_[trainIdx] = queryIdx;
    }
}

// Mark III
void Tester::triangulate3(const VisualOdometry &visual_odometry,
                            cv::Mat camera_Matrix,
                            std::shared_ptr<Frame> &pPrev_frame,
                            std::shared_ptr<Frame> &pCurr_frame,
                            const std::vector<TestMatch> good_matches,
                            const cv::Mat &mask,
                            std::vector<Eigen::Vector3d> &frame_keypoints_3d) {
    std::cout << "----- Tester::triangulate3 -----" << std::endl;

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
                landmark_frame_id = pPrevNFrame->id_;
                landmark_id = pPrevNFrame->keypoint_landmark_[kp_idx].second;
                found_landmark = (landmark_id != -1);

                if (found_landmark) {
                    break;
                }

                kp_idx = pPrevNFrame->matches_with_prev_frame_[kp_idx];
                pPrevNFrame = pPrevNFrame->pPrevious_frame_.lock();
            }
            if (found_landmark) {
                new_landmark = false;
                landmark_accepted = true;
            }

            // // hard matching
            // if (pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].second != -1) {  // landmark already exists
            //     new_landmark = false;

            //     int landmark_idx = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].first;
            //     auto pLandmark = pPrev_frame->landmarks_[landmark_idx];


            //     // TEST mode에서는 모든 feature 쌍이 올바르게 매칭되었다고 가정.
            //     landmark_accepted = true;
            // }

            // execute triangulation
            Eigen::Vector3d w_keypoint_3d;  // keypoint coordinate in world frame
            visual_odometry.pUtils_->triangulateKeyPoint(pCurr_frame, prev_frame_kp_pt, curr_frame_kp_pt, w_keypoint_3d);

            if (new_landmark) {
                std::shared_ptr<Landmark> pLandmark = std::make_shared<Landmark>();
                pLandmark->observations_.insert({pPrev_frame->id_, good_matches[i].queryIdx});
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});
                pLandmark->point_3d_ = w_keypoint_3d;

                pPrev_frame->landmarks_.push_back(pLandmark);
                pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx] = std::pair(pPrev_frame->landmarks_.size() - 1, pLandmark->id_);
                // pPrev_frame->keypoints_3d_[good_matches[i].queryIdx] = w_keypoint_3d;
                pCurr_frame->landmarks_.push_back(pLandmark);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, pLandmark->id_);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                frame_keypoints_3d.push_back(w_keypoint_3d);
            }
            else if (landmark_accepted) {
                // // add information to curr_frame
                // std::pair<int, int> prev_frame_kp_lm = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx];
                // int landmark_id = prev_frame_kp_lm.second;
                // pCurr_frame->landmarks_.push_back(pPrev_frame->landmarks_[prev_frame_kp_lm.first]);
                // pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size(), landmark_id);
                // pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                std::pair<int, int> prev_n_frame_kp_lm = pPrevNFrame->keypoint_landmark_[kp_idx];
                pCurr_frame->landmarks_.push_back(pPrevNFrame->landmarks_[prev_n_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size() - 1, landmark_id);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrevNFrame->landmarks_[prev_n_frame_kp_lm.first];
                pLandmark->observations_.insert({pCurr_frame->id_, good_matches[i].trainIdx});

                frame_keypoints_3d.push_back(w_keypoint_3d);
            }
        }
    }
}

void Tester::drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &good_matches,
                                    // const cv::Mat &essential_mask,
                                    const cv::Mat &pose_mask,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pFrame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, good_matches, cv::Mat(), triangulated_kps, prev_projected_pts, curr_projected_pts);

    int essential_inlier_cnt = 0;
    int pose_inlier_cnt = 0;
    for (int i = 0; i < good_matches.size(); i++) {
        // if (essential_mask.at<unsigned char>(i) != 1) {
        //     continue;
        // }

        cv::Point2f measurement_point0 = pPrev_frame->keypoints_pt_[good_matches[i].queryIdx];
        cv::Point2f measurement_point1 = pFrame->keypoints_pt_[good_matches[i].trainIdx];

        // draw images
        cv::rectangle(prev_frame_img,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 127, 255));  // green, (orange)
        cv::circle(prev_frame_img,
                    prev_projected_pts[i],
                    2,
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(prev_frame_img,
                    measurement_point0,
                    prev_projected_pts[i],
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::rectangle(curr_frame_img,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 127, 255));  // green, (orange)
        cv::circle(curr_frame_img,
                    curr_projected_pts[i],
                    2,
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(curr_frame_img,
                    measurement_point1,
                    curr_projected_pts[i],
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)

        essential_inlier_cnt++;
        if (pose_mask.at<unsigned char>(i) == 1) {
            pose_inlier_cnt++;
        }
    }
    cv::Mat result_image;
    cv::hconcat(prev_frame_img, curr_frame_img, result_image);

    cv::putText(result_image, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pFrame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(result_image, "#essential inliers / matched features: " + std::to_string(essential_inlier_cnt) + " / " + std::to_string(good_matches.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(result_image, "#recover pose inliers: " + std::to_string(pose_inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/inter_frames/reprojected_landmarks/frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + "frame" + std::to_string(pFrame->frame_image_idx_) + "_proj.png", result_image);
}


void Tester::reprojectLandmarks(const std::shared_ptr<Frame> &pFrame,
                            const std::vector<TestMatch> &matches,
                            const cv::Mat &mask,
                            const std::vector<Eigen::Vector3d> &landmark_points_3d,
                            std::vector<cv::Point2f> &prev_projected_pts,
                            std::vector<cv::Point2f> &curr_projected_pts) {
    // std::cout << "----- Utils::reprojectLandmarks -----" << std::endl;
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    Eigen::Matrix3d K;
    cv::cv2eigen(pFrame->pCamera_->intrinsic_, K);

    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    // prev_proj = K * prev_proj * pPrev_frame->pose_.matrix();
    // curr_proj = K * curr_proj * pFrame->pose_.matrix();
    prev_proj = K * prev_proj * pPrev_frame->pose_.inverse().matrix();
    curr_proj = K * curr_proj * pFrame->pose_.inverse().matrix();

    // std::cout << "prev pose:\n" << pPrev_frame->pose_.matrix() << std::endl;
    // std::cout << "curr pose:\n" << pFrame->pose_.matrix() << std::endl;

    for (int i = 0; i < matches.size(); i++) {
        if (mask.empty() || mask.at<unsigned char>(i) == 1) {
            Eigen::Vector3d landmark_point_3d = landmark_points_3d[i];
            Eigen::Vector4d landmark_point_3d_homo(landmark_point_3d[0],
                                                    landmark_point_3d[1],
                                                    landmark_point_3d[2],
                                                    1);

            Eigen::Vector3d img0_x_tilde = prev_proj * landmark_point_3d_homo;
            Eigen::Vector3d img1_x_tilde = curr_proj * landmark_point_3d_homo;

            cv::Point2f projected_point0(img0_x_tilde[0] / img0_x_tilde[2],
                                        img0_x_tilde[1] / img0_x_tilde[2]);
            cv::Point2f projected_point1(img1_x_tilde[0] / img1_x_tilde[2],
                                        img1_x_tilde[1] / img1_x_tilde[2]);

            prev_projected_pts.push_back(projected_point0);
            curr_projected_pts.push_back(projected_point1);
        }
    }
}

double Tester::calcReprojectionError(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &matches,
                                    const cv::Mat &mask,
                                    const std::vector<Eigen::Vector3d> &landmark_points_3d) {
    // assume matches.size() = landmark_points_3d.size()
    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, matches, cv::Mat(), landmark_points_3d, prev_projected_pts, curr_projected_pts);

    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    double error_prev = 0, error_curr = 0;
    double reproj_error_prev = 0, reproj_error_curr = 0;
    int point_cnt = 0;
    for (int i = 0; i < landmark_points_3d.size(); i++) {
        if (mask.at<unsigned char>(i) == 1) {
            // cv::Point2f measurement_point0 = pPrev_frame->keypoints_[matches[i].queryIdx].pt;
            // cv::Point2f measurement_point1 = pFrame->keypoints_[matches[i].trainIdx].pt;
            cv::Point2f measurement_point0 = pPrev_frame->keypoints_pt_[matches[i].queryIdx];
            cv::Point2f measurement_point1 = pFrame->keypoints_pt_[matches[i].trainIdx];

            cv::Point2f error_prev_pt = (prev_projected_pts[i] - measurement_point0);
            cv::Point2f error_curr_pt = (curr_projected_pts[i] - measurement_point1);

            error_prev = sqrt(error_prev_pt.x * error_prev_pt.x + error_prev_pt.y * error_prev_pt.y);
            error_curr = sqrt(error_curr_pt.x * error_curr_pt.x + error_curr_pt.y * error_curr_pt.y);

            reproj_error_prev += error_prev;
            reproj_error_curr += error_curr;

            point_cnt++;
        }
    }

    double prev_error_mean, curr_error_mean;
    prev_error_mean = reproj_error_prev / point_cnt;
    curr_error_mean = reproj_error_curr / point_cnt;
    std::cout << "prev_error_mean: " << prev_error_mean << std::endl;
    std::cout << "curr_error_mean: " << curr_error_mean << std::endl;

    double reprojection_error = ((reproj_error_prev + reproj_error_curr) / point_cnt) / 2;
    return reprojection_error;
}

/* // VisualOdometry::triangulate3
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
                    // orb_matcher_->match(new_descriptors, target_descriptors, match_new_target);
                    // orb_matcher_->match(target_descriptors, new_descriptors, match_target_new);
                    sift_matcher_->match(new_descriptors, target_descriptors, match_new_target);
                    sift_matcher_->match(target_descriptors, new_descriptors, match_target_new);
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
}*/






