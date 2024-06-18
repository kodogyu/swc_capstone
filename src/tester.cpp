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

    std::cout << "frame [0] image: " << vo.pConfig_->left_image_entries_[0] << std::endl;
    image0_left = cv::imread(vo.pConfig_->left_image_entries_[0], cv::IMREAD_GRAYSCALE);

    std::shared_ptr<Frame> pPrev_frame = std::make_shared<Frame>();
    pPrev_frame->image_ = image0_left;
    pPrev_frame->frame_image_idx_ = vo.pConfig_->frame_offset_;
    pPrev_frame->pCamera_ = vo.pCamera_;
    pPrev_frame->pose_ = Eigen::Isometry3d::Identity();
    vo.poses_.push_back(pPrev_frame->pose_);
    vo.frames_.push_back(pPrev_frame);
    vo.frame_window_.push_back(pPrev_frame);

    for (int run_iter = 1; run_iter < vo.pConfig_->num_frames_; run_iter++) {
        // start timer [total time cost]
        Timer total_timer;
        total_timer.start();

        std::cout << "frame [" << run_iter << "] image: " << vo.pConfig_->left_image_entries_[run_iter] << std::endl;
        image1_left = cv::imread(vo.pConfig_->left_image_entries_[run_iter], cv::IMREAD_GRAYSCALE);

        // new Frame!
        std::shared_ptr<Frame> pCurr_frame = std::make_shared<Frame>();
        pCurr_frame->image_ = image1_left;
        pCurr_frame->frame_image_idx_ = vo.pConfig_->frame_offset_ + run_iter;
        pCurr_frame->pCamera_ = vo.pCamera_;
        pCurr_frame->pPrevious_frame_ = pPrev_frame;
        pPrev_frame->pNext_frame_ = pCurr_frame;

        //**========== 1. Feature extraction ==========**//
        std::cout << "\n1. Feature extraction" << std::endl;

        Timer feature_timer;
        feature_timer.start();

        //* TEST mode

        // 1. Checkerboard feature
        /*
        cv::Size checkerboard_size = cv::Size(6, 8);
        std::vector<cv::Size> checkerboard_sizes = {cv::Size(15, 10), cv::Size(10, 7), cv::Size(9, 6), cv::Size(7, 4), cv::Size(4, 3)};

        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature extraction" << std::endl;

            cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001 );

            if (run_iter == 1) {  // first run
                // // --- handmade features
                // setFrameKeypoints_pt(pPrev_frame, manual_kp_frame0_);

                // // --- checkerboard points
                // std::vector<cv::Point2f> prev_corners;
                // cv::findChessboardCorners(pPrev_frame->image_, checkerboard_size, prev_corners);
                // cv::cornerSubPix(pPrev_frame->image_, prev_corners, cv::Size(11, 11), cv::Size(-1,-1), criteria);
                // pPrev_frame->keypoints_pt_ = prev_corners;
                // setFrameKeypoints_pt(pPrev_frame, prev_corners);

                // --- multiple checkerboard points
                std::vector<std::vector<cv::Point2f>> prev_corners_vec;
                prev_corners_vec = findMultipleCheckerboards(pPrev_frame->image_, checkerboard_sizes, 1);
                setFrameKeypoints_pt(pPrev_frame, prev_corners_vec);

                vo.pUtils_->drawKeypoints(pPrev_frame);
            }
            // // --- handmade features
            // setFrameKeypoints_pt(pCurr_frame, manual_kps_vec_[run_iter]);

            // // --- checkerboard points
            // std::vector<cv::Point2f> curr_corners;
            // cv::findChessboardCorners(pCurr_frame->image_, checkerboard_size, curr_corners);
            // cv::cornerSubPix(pCurr_frame->image_, curr_corners, cv::Size(11, 11), cv::Size(-1,-1), criteria);
            // pCurr_frame->keypoints_pt_ = curr_corners;
            // setFrameKeypoints_pt(pCurr_frame, curr_corners);

            // --- multiple checkerboard points
            std::vector<std::vector<cv::Point2f>> curr_corners_vec;
            curr_corners_vec = findMultipleCheckerboards(pCurr_frame->image_, checkerboard_sizes, 1);
            setFrameKeypoints_pt(pCurr_frame, curr_corners_vec);

            vo.pUtils_->drawKeypoints(pCurr_frame);
        }
        */

        // 2. ORB-SLAM3 features, and matches
        std::string feature_match_dir = "/home/kodogyu/github_repos/ORB_SLAM3/my_dir/";

        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature extraction" << std::endl;

            if (run_iter == 1) {  // first run
                // --- ORB-SLAM3 features
                setFrameKeypoints_pt(pPrev_frame, readKeypoints(feature_match_dir, pPrev_frame->id_, vo.pConfig_->frame_offset_));

                vo.pUtils_->drawKeypoints(pPrev_frame);
            }
            // --- ORB-SLAM3 features
            setFrameKeypoints_pt(pCurr_frame, readKeypoints(feature_match_dir, pCurr_frame->id_, vo.pConfig_->frame_offset_));

            vo.pUtils_->drawKeypoints(pCurr_frame);
        }

        vo.feature_extraction_costs_.push_back(feature_timer.stop());

        //**========== 2. Feature matching ==========**//
        std::cout << "\n2. Feature matching" << std::endl;

        // start timer [feature matching]
        Timer feature_matching_timer;
        feature_matching_timer.start();

        std::vector<TestMatch> good_matches_test;  // for TEST mode
        // std::vector<cv::DMatch> good_matches;  // good matchings

        int raw_matches_size = -1;
        int good_matches_size = -1;

        // extract points from keypoints
        std::vector<cv::Point2f> image0_kp_pts;
        std::vector<cv::Point2f> image1_kp_pts;

        //* TEST mode

        // 1. checkerboard matches
        /*
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature matching" << std::endl;

            // // handmade feature matches
            // good_matches_test = manual_matches_vec_[run_iter - 1];
            // raw_matches_size = manual_matches_vec_[run_iter - 1].size();

            // // single checkerboard matches
            // std::cout << "checkerboard corners " << std::endl;
            // for (int i = 0; i < checkerboard_size.area(); i++) {
            //     good_matches_test.push_back(TestMatch(i, i));
            // }

            // multiple checkerboard matches
            int kp_idx = 0;
            for (int pattern_idx = 0; pattern_idx < checkerboard_sizes.size(); pattern_idx++) {
                if (pPrev_frame->keypoints_pt_[kp_idx] == cv::Point2f(-1, -1) || pCurr_frame->keypoints_pt_[kp_idx] == cv::Point2f(-1, -1)) {
                    ;  // do nothing
                }
                else {
                    for (int i = 0; i < checkerboard_sizes[pattern_idx].area(); i++) {
                        good_matches_test.push_back(TestMatch(kp_idx + i, kp_idx + i));
                    }
                }

                kp_idx += checkerboard_sizes[pattern_idx].area();
            }

            raw_matches_size = good_matches_test.size();


            for (auto match : good_matches_test) {
                image0_kp_pts.push_back(pPrev_frame->keypoints_pt_[match.queryIdx]);
                image1_kp_pts.push_back(pCurr_frame->keypoints_pt_[match.trainIdx]);
            }

            // set keyframe matches
            setFrameMatches(pCurr_frame, good_matches_test);

            good_matches_size = good_matches_test.size();
        }
        */

        // 2. ORB-SLAM3 features, and matches
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Feature matching" << std::endl;

            // handmade feature matches
            good_matches_test = readMatches(feature_match_dir, pPrev_frame->id_, vo.pConfig_->frame_offset_);
            raw_matches_size = manual_matches_vec_[run_iter - 1].size();

            raw_matches_size = good_matches_test.size();

            // filter matches
            if (vo.pConfig_->filtering_mode_ == FilterMode::MATCH_FILTERING) {
                std::cout << "match size before filtering: " << good_matches_test.size() << std::endl;
                filterMatches(pPrev_frame, good_matches_test, vo.pConfig_);
                std::cout << "match size after filtering: " << good_matches_test.size() << std::endl;

                // draw grid
                vo.pUtils_->drawGrid(pPrev_frame->image_);

                // draw keypoints
                std::string tail;
                tail = "_(" + std::to_string(vo.pConfig_->patch_width_) + ", " + std::to_string(vo.pConfig_->patch_height_) + ", " + std::to_string(vo.pConfig_->kps_per_patch_) + ")";
                vo.pUtils_->drawKeypoints(pPrev_frame, "output_logs/filtered_matches", tail);
            }

            // set keyframe matches
            setFrameMatches(pCurr_frame, good_matches_test);

            good_matches_size = good_matches_test.size();

            for (auto match : good_matches_test) {
                image0_kp_pts.push_back(pPrev_frame->keypoints_pt_[match.queryIdx]);
                image1_kp_pts.push_back(pCurr_frame->keypoints_pt_[match.trainIdx]);
            }
        }

        std::cout << "original features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << raw_matches_size << std::endl;
        std::cout << "good features for image" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_) + ": " << good_matches_size << std::endl;

        // essential matrix
        cv::Mat essential_mat = cv::Mat::zeros(3, 3, CV_32F);
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
        vo.feature_matching_costs_.push_back(feature_matching_timer.stop());

        // draw matches
        drawMatches(pPrev_frame, pCurr_frame, good_matches_test);

        //**========== 3. Motion estimation ==========**//
        std::cout << "\n3. Motion estimation" << std::endl;

        // start timer [motion estimation]
        Timer motion_estimation_timer;
        motion_estimation_timer.start();

        Eigen::Isometry3d relative_pose;
        cv::Mat pose_mask = essential_mask.clone();

        //* TEST mode

        // 1. checkerboard motion estimation
        /*
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Motion estimation" << std::endl;

            // // --- My implementation
            // vo.pUtils_->recoverPose(vo.pCamera_->intrinsic_, essential_mat, image0_kp_pts, image1_kp_pts, relative_pose, pose_mask);

            // // --- GT relative pose
            // relative_pose = vo.pUtils_->getGT(pPrev_frame->frame_image_idx_).inverse() * vo.pUtils_->getGT(pCurr_frame->frame_image_idx_);

            if (run_iter == 1) {
                // --- OpenCV functions (recoverPose)
                cv::Mat R, t;
                cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, pCurr_frame->pCamera_->intrinsic_, R, t, pose_mask);
                Eigen::Matrix3d rotation_mat;
                Eigen::Vector3d translation_mat;
                cv::cv2eigen(R, rotation_mat);
                cv::cv2eigen(t, translation_mat);
                relative_pose.linear() = rotation_mat;
                relative_pose.translation() = translation_mat;
                relative_pose = relative_pose.inverse();
            }
            else {
                // --- OpenCV functions (solvePnP)
                cv::Mat rvec, tvec;
                std::vector<cv::Point3d> object_points;
                std::vector<cv::Point2d> image_points;

                for (int i = 0; i < good_matches_size; i++) {
                // for (int i = 0; i < checkerboard_sizes[0].area(); i++) {
                    // object points
                    Eigen::Vector3d object_point;

                    //! 해당 랜드마크에서 다른 frame의 keypoint와 유사성도 고려해야 함.
                    int landmark_idx = pPrev_frame->keypoint_landmark_.at(good_matches_test[i].queryIdx).first;
                    object_point = pPrev_frame->landmarks_.at(landmark_idx)->point_3d_;

                    cv::Point3d object_point_cv(object_point.x(), object_point.y(), object_point.z());
                    object_points.push_back(object_point_cv);

                    // image points
                    cv::Point2f keypoint_pt = pCurr_frame->keypoints_pt_.at(good_matches_test[i].trainIdx);
                    image_points.push_back(keypoint_pt);

                    std::cout << "landmark idx: " << landmark_idx << ", image1 keypoint idx: " << good_matches_test[i].trainIdx << std::endl;
                }

                bool pnp_result = cv::solvePnP(object_points, image_points, vo.pCamera_->intrinsic_, vo.pCamera_->distortion_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                // cv::solvePnP(object_points, image_points, vo.pCamera_->intrinsic_, vo.pCamera_->distortion_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
                // cv::solvePnP(object_points, image_points, vo.pCamera_->intrinsic_, vo.pCamera_->distortion_, rvec, tvec, false, cv::SOLVEPNP_EPNP);
                std::cout << "pnp result: " << pnp_result << std::endl;
                std::cout << "rvec.size: " << rvec.size << std::endl;
                std::cout << "tvec.size: " << tvec.size << std::endl;

                cv::Mat R, t;
                cv::Rodrigues(rvec, R);
                t = tvec;

                Eigen::Matrix3d rotation_mat;
                Eigen::Vector3d translation_mat;
                cv::cv2eigen(R, rotation_mat);
                cv::cv2eigen(t, translation_mat);
                relative_pose.linear() = rotation_mat;
                relative_pose.translation() = translation_mat;
                relative_pose = relative_pose.inverse();
            }
        }
        */

        // 2. general pose estimation
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Motion estimation" << std::endl;

            // --- My implementation
            vo.pUtils_->recoverPose(vo.pCamera_->intrinsic_, essential_mat, image0_kp_pts, image1_kp_pts, relative_pose, pose_mask);
        }
        std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;

        pCurr_frame->relative_pose_ = relative_pose;
        pCurr_frame->pose_ = pPrev_frame->pose_ * relative_pose;
        vo.poses_.push_back(pCurr_frame->pose_);
        vo.relative_poses_.push_back(relative_pose);

        int pose_mask_cnt = vo.pUtils_->countMask(pose_mask);
        std::cout << "pose_mask cnt: " << pose_mask_cnt << std::endl;

        // end timer [motion estimation]
        vo.motion_estimation_costs_.push_back(motion_estimation_timer.stop());

        //**========== 4. Triangulation ==========**//
        std::cout << "\n4. Triangulation" << std::endl;

        // start timer [triangulation]
        Timer triangulation_timer;
        triangulation_timer.start();


        std::vector<Eigen::Vector3d> keypoints_3d;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            std::cout << "\n[test mode] Triangulation" << std::endl;

            //! essential_mask was pose_mask
            // triangulate3(vo, vo.pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches_test, essential_mask, keypoints_3d);
            triangulate3(vo, vo.pCamera_->intrinsic_, pPrev_frame, pCurr_frame, good_matches_test, pose_mask, keypoints_3d);
            std::cout << "[debug] Triangulation done." << std::endl;
        }

        // end timer [triangulation]
        vo.triangulation_costs_.push_back(triangulation_timer.stop());

        // ----- Calculate & Draw reprojection error
        std::vector<Eigen::Vector3d> triangulated_kps, triangulated_kps_cv;

        //* TEST mode
        if (vo.pConfig_->test_mode_) {
            vo.pUtils_->triangulateKeyPoints(pCurr_frame, image0_kp_pts, image1_kp_pts, triangulated_kps);
            drawReprojectedKeypoints3D(pCurr_frame, good_matches_test, pose_mask, triangulated_kps);
            drawReprojectedLandmarks(pCurr_frame, good_matches_test);
            if (vo.pConfig_->calc_reprojection_error_) {
                double reprojection_error_kps = calcReprojectionError(pCurr_frame, good_matches_test, pose_mask, triangulated_kps);
                std::cout << "triangulated kps reprojection error: " << reprojection_error_kps << std::endl;

                double reprojection_error_landmarks = vo.pUtils_->calcReprojectionError(pCurr_frame);
                std::cout << "landmarks reprojection error: " << reprojection_error_landmarks << std::endl;
            }
        }

        //**========== 5. Scale estimation ==========**//
        std::cout << "\n5. Scale estimation" << std::endl;
        // start timer [scaling]
        Timer scaling_timer;
        scaling_timer.start();

        // std::vector<int> scale_mask(pCurr_frame->keypoints_pt_.size(), 0);
        // double est_scale_ratio = estimateScale(vo, pPrev_frame, pCurr_frame, scale_mask);
        // double gt_scale_ratio = vo.getGTScale(pCurr_frame);
        // std::cout << "estimated scale: " << est_scale_ratio << ". GT scale: " << gt_scale_ratio << std::endl;
        // vo.scales_.push_back(est_scale_ratio);
        // vo.gt_scales_.push_back(gt_scale_ratio);
        // applyScale(pCurr_frame, gt_scale_ratio, scale_mask);
        // applyScale(pCurr_frame, est_scale_ratio, scale_mask);

        // end timer [scaling]
        vo.scaling_costs_.push_back(scaling_timer.stop());

        // cv::Mat keypoints_3d_mat = cv::Mat(3, pPrev_frame->keypoints_3d_.size(), CV_64F);
        // for (int i = 0; i < keypoints_3d.size(); i++) {
        //     keypoints_3d_mat.at<double>(0, i) = pPrev_frame->keypoints_3d_[i].x();
        //     keypoints_3d_mat.at<double>(1, i) = pPrev_frame->keypoints_3d_[i].y();
        //     keypoints_3d_mat.at<double>(2, i) = pPrev_frame->keypoints_3d_[i].z();
        // }
        // keypoints_3d_vec_.push_back(keypoints_3d_mat);

        //**========== 6. Local optimization ==========**//
        std::cout << "\n6. Local optimization" << std::endl;

        // start timer [optimization]
        Timer optimization_timer;
        optimization_timer.start();


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
        vo.optimization_costs_.push_back(optimization_timer.stop());

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
        vo.total_time_costs_.push_back(total_timer.stop());

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

void Tester::setFrameKeypoints_pt(const std::shared_ptr<Frame> &pFrame, std::vector<std::vector<cv::Point2f>> kp_pts_vec) {
    std::cout << "----- Tester::setFrameKeypoints_pt -----" << std::endl;

    int total_kp_pts_cnt = 0;

    // init keypoints_pt_
    for (auto kp_pts : kp_pts_vec) {
        pFrame->keypoints_pt_.insert(pFrame->keypoints_pt_.end(), kp_pts.begin(), kp_pts.end());

        total_kp_pts_cnt += kp_pts.size();
    }

    // init 3D keypoints
    pFrame->keypoints_3d_.assign(total_kp_pts_cnt, Eigen::Vector3d(0.0, 0.0, 0.0));

    // init matches
    pFrame->matches_with_prev_frame_ = std::vector<int>(total_kp_pts_cnt, -1);
    pFrame->matches_with_next_frame_ = std::vector<int>(total_kp_pts_cnt, -1);

    // init depths_
    pFrame->depths_.reserve(total_kp_pts_cnt);

    // init keypoint_landmark_
    pFrame->keypoint_landmark_.assign(total_kp_pts_cnt, std::pair(-1, -1));
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
            cv::Point2f prev_frame_kp_pt = pPrev_frame->keypoints_pt_.at(good_matches[i].queryIdx);
            cv::Point2f curr_frame_kp_pt = pCurr_frame->keypoints_pt_.at(good_matches[i].trainIdx);

            // keypoint가 이전 프레임에서 이제까지 한번도 landmark로 선택되지 않았는지 확인 (보존하고 있는 frame에 한해서 검사)
            // bool found_landmark = false;
            int kp_idx = good_matches[i].queryIdx;
            std::shared_ptr<Frame> pPrevNFrame = pPrev_frame;
            // int landmark_frame_id = -1;
            int landmark_id = -1;
            // while (pPrevNFrame->id_ > 0) {
            //     landmark_frame_id = pPrevNFrame->id_;
            //     landmark_id = pPrevNFrame->keypoint_landmark_.at(kp_idx).second;
            //     found_landmark = (landmark_id != -1);

            //     if (found_landmark) {
            //         break;
            //     }

            //     kp_idx = pPrevNFrame->matches_with_prev_frame_.at(kp_idx);

            //     if (kp_idx == -1) {
            //         break;
            //     }
            //     pPrevNFrame = pPrevNFrame->pPrevious_frame_.lock();
            // }
            // if (found_landmark) {
            //     new_landmark = false;
            //     landmark_accepted = true;
            // }

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

void Tester::drawReprojectedKeypoints3D(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &good_matches,
                                    const cv::Mat &pose_mask,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pFrame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, cv::Mat(), triangulated_kps, prev_projected_pts, curr_projected_pts);

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

    cv::imwrite("output_logs/inter_frames/reprojected_landmarks/frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + "frame" + std::to_string(pFrame->frame_image_idx_) + "_proj_kps.png", result_image);
}

void Tester::drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<TestMatch> &good_matches) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pFrame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);

    std::vector<Eigen::Vector3d> landmarks;
    for (auto pLandmark : pFrame->landmarks_) {
        landmarks.push_back(pLandmark->point_3d_);
    }

    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, cv::Mat(), landmarks, prev_projected_pts, curr_projected_pts);

    int essential_inlier_cnt = 0;
    // for (int i = 0; i < good_matches.size(); i++) {
    for (int i = 0; i < landmarks.size(); i++) {
        std::shared_ptr<Landmark> pLandmark = pFrame->landmarks_[i];

        int prev_frame_kp_idx = pLandmark->observations_.find(pPrev_frame->id_)->second;
        int curr_frame_kp_idx = pLandmark->observations_.find(pFrame->id_)->second;
        cv::Point2f measurement_point0 = pPrev_frame->keypoints_pt_[prev_frame_kp_idx];
        cv::Point2f measurement_point1 = pFrame->keypoints_pt_[curr_frame_kp_idx];

        // draw images
        cv::rectangle(prev_frame_img,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (orange)
        cv::circle(prev_frame_img,
                    prev_projected_pts[i],
                    2,
                    cv::Scalar(0, 0, 255));  // red, (blue)
        cv::line(prev_frame_img,
                    measurement_point0,
                    prev_projected_pts[i],
                    cv::Scalar(0, 0, 255));  // red, (blue)
        cv::rectangle(curr_frame_img,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (orange)
        cv::circle(curr_frame_img,
                    curr_projected_pts[i],
                    2,
                    cv::Scalar(0, 0, 255));  // red, (blue)
        cv::line(curr_frame_img,
                    measurement_point1,
                    curr_projected_pts[i],
                    cv::Scalar(0, 0, 255));  // red, (blue)

        essential_inlier_cnt++;
    }
    cv::Mat result_image;
    cv::hconcat(prev_frame_img, curr_frame_img, result_image);

    cv::putText(result_image, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pFrame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(result_image, "#essential inliers / matched features: " + std::to_string(essential_inlier_cnt) + " / " + std::to_string(good_matches.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/inter_frames/reprojected_landmarks/frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + "frame" + std::to_string(pFrame->frame_image_idx_) + "_proj_landmarks.png", result_image);
}

void Tester::drawMatches(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, const std::vector<TestMatch> &good_matches) {
    cv::Mat prev_frame_img, curr_frame_img;
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pCurr_frame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);

    cv::Mat image_matches;
    cv::hconcat(prev_frame_img, curr_frame_img, image_matches);

    int image_w = prev_frame_img.cols;
    // int image_h = prev_frame_img.rows;

    for (int i = 0; i < good_matches.size(); i++) {
        cv::Point2f prev_kp_pt = pPrev_frame->keypoints_pt_[good_matches[i].queryIdx];
        cv::Point2f curr_kp_pt = pCurr_frame->keypoints_pt_[good_matches[i].trainIdx];

        curr_kp_pt.x += image_w;

        // draw previous frame keypoint
        cv::rectangle(image_matches,
                    prev_kp_pt - cv::Point2f(5, 5),
                    prev_kp_pt + cv::Point2f(5, 5),
                    cv::Scalar(0, 127, 255));  // orange

        // draw current frame keypoint
        cv::rectangle(image_matches,
                    curr_kp_pt - cv::Point2f(5, 5),
                    curr_kp_pt + cv::Point2f(5, 5),
                    cv::Scalar(0, 127, 255));  // orange

        // draw prev-current keypoint line
        cv::line(image_matches,
                    prev_kp_pt,
                    curr_kp_pt,
                    cv::Scalar(0, 255, 0));  // green
    }
    cv::putText(image_matches, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/inter_frames/frame"
                + std::to_string(pPrev_frame->frame_image_idx_)
                + "&"
                + "frame"
                + std::to_string(pCurr_frame->frame_image_idx_)
                + "_kp_matches(raw).png", image_matches);
}


void Tester::reprojectLandmarks(const std::shared_ptr<Frame> &pFrame,
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

    for (int i = 0; i < landmark_points_3d.size(); i++) {
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
    reprojectLandmarks(pFrame, cv::Mat(), landmark_points_3d, prev_projected_pts, curr_projected_pts);

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


double Tester::estimateScale(VisualOdometry &vo,const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, std::vector<int> &scale_mask) {
    std::cout << "----- Tester::estimateScale -----" << std::endl;
    double scale_ratio = 1.0;

    std::vector<int> prev_frame_covisible_feature_idxs, curr_frame_covisible_feature_idxs;

    if (pPrev_frame->id_ == 0) {
        return 1.0;
    }

    std::shared_ptr<Frame> pBefore_prev_frame = pPrev_frame->pPrevious_frame_.lock();
    
    //**========== 2. Feature matching ==========**//
    std::vector<cv::DMatch> good_matches;  // good matchings

    int raw_matches_size = -1;
    int good_matches_size = -1;

    // extract points from keypoints
    std::vector<cv::Point2f> image0_kp_pts;
    std::vector<cv::Point2f> image2_kp_pts;

    // image0 & image1 (matcher matching)
    std::vector<std::vector<cv::DMatch>> image_matches02_vec;
    std::vector<std::vector<cv::DMatch>> image_matches20_vec;
    vo.knnMatch(pBefore_prev_frame->descriptors_, pCurr_frame->descriptors_, image_matches02_vec, 2);  // beforePrev -> curr matches
    vo.knnMatch(pCurr_frame->descriptors_, pPrev_frame->descriptors_, image_matches20_vec, 2);  // curr -> beforePrev matches

    raw_matches_size = image_matches02_vec.size();

    // Mark II
    for (int i = 0; i < image_matches02_vec.size(); i++) {
        if (image_matches02_vec[i][0].distance < image_matches02_vec[i][1].distance * vo.pConfig_->des_dist_thresh_) {  // prev -> curr match에서 좋은가?
            int image1_keypoint_idx = image_matches02_vec[i][0].trainIdx;
            if (image_matches20_vec[image1_keypoint_idx][0].distance < image_matches20_vec[image1_keypoint_idx][1].distance * vo.pConfig_->des_dist_thresh_) {  // curr -> prev match에서 좋은가?
                if (image_matches02_vec[i][0].queryIdx == image_matches20_vec[image1_keypoint_idx][0].trainIdx)
                    good_matches.push_back(image_matches02_vec[i][0]);
            }
        }
    }

    good_matches_size = good_matches.size();

    // covisible features
    std::vector<int> covisible_feature_idxs_02_0, covisible_feature_idxs_01_1, covisible_feature_idxs_02_2;

    for (int i = 0; i < pBefore_prev_frame->matches_with_next_frame_.size(); i++) {
        int pBefore_prev_kp_idx = i;
        for (int j = 0; j < good_matches.size(); j++) {
            if (good_matches[j].queryIdx == pBefore_prev_kp_idx) {
                covisible_feature_idxs_02_0.push_back(good_matches[j].queryIdx);
                covisible_feature_idxs_02_2.push_back(good_matches[j].trainIdx);
                covisible_feature_idxs_01_1.push_back(pBefore_prev_frame->matches_with_next_frame_.at(i));
            }
        }
    }

    std::vector<cv::Point2f> frame0_kp_pts, frame1_kp_pts, frame2_kp_pts;
    for (int i = 0; i < covisible_feature_idxs_02_0.size(); i++) {
        frame0_kp_pts.push_back(pBefore_prev_frame->keypoints_pt_.at(covisible_feature_idxs_02_0[i]));
        frame1_kp_pts.push_back(pPrev_frame->keypoints_pt_.at(covisible_feature_idxs_01_1[i]));
        frame2_kp_pts.push_back(pCurr_frame->keypoints_pt_.at(covisible_feature_idxs_02_2[i]));

        std::cout << "frame0_kp_pts[" << i << "] " << frame0_kp_pts[i] << std::endl;
        std::cout << "frame1_kp_pts[" << i << "] " << frame1_kp_pts[i] << std::endl;
        std::cout << "frame2_kp_pts[" << i << "] " << frame2_kp_pts[i] << std::endl;
    }


    // triangulate
    std::vector<Eigen::Vector3d> keypoints3d_01, keypoints3d_02;
    vo.pUtils_->triangulateKeyPoints(pPrev_frame, frame0_kp_pts, frame1_kp_pts, keypoints3d_01);
    vo.pUtils_->triangulateKeyPoints(pCurr_frame, frame0_kp_pts, frame2_kp_pts, keypoints3d_02);

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

    // scale_ratio = curr_frame_landmark_distance / prev_frame_landmark_distance;
    scale_ratio = acc_distance_02 / acc_distance_01;

    return scale_ratio;
}

std::vector<cv::Point2f> Tester::findMultipleCheckerboards(const cv::Mat &image, const cv::Size &patternSize, int nCheckerboards) {
    std::vector<cv::Point2f> total_corners;

    cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001 );
    cv::Mat image_copy = image.clone();

    // draw checkerboard corners
    cv::Mat draw_image;
    cv::cvtColor(image, draw_image, cv::COLOR_GRAY2BGR);
    static int num = 0;

    for (int n = 0; n < nCheckerboards; n++) {
        std::vector<cv::Point2f> corners;

        cv::findChessboardCorners(image_copy, patternSize, corners);
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1,-1), criteria);

        bool found;
        cv::drawChessboardCorners(draw_image, patternSize, corners, found);

        // hide detected region
        cv::Point2f top_left = corners[0] - cv::Point2f(20, 20);
        cv::Point2f bottom_right = corners[patternSize.area() - 1] + cv::Point2f(20, 20);
        cv::rectangle(image_copy, cv::Rect(top_left, bottom_right), cv::Scalar(0, 0, 0), cv::FILLED);

        total_corners.insert(total_corners.end(), corners.begin(), corners.end());
    }


    cv::imwrite("output_logs/landmarks/frame" + std::to_string(num) + "_checkboard_corners.png", draw_image);
    num++;

    return total_corners;
}

std::vector<std::vector<cv::Point2f>> Tester::findMultipleCheckerboards(const cv::Mat &image, const std::vector<cv::Size> &patternSizes, int nCheckerboardsEach) {
    std::vector<std::vector<cv::Point2f>> total_corners(patternSizes.size());

    cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001 );
    cv::Mat image_copy = image.clone();

    // draw checkerboard corners
    cv::Mat draw_image;
    cv::cvtColor(image, draw_image, cv::COLOR_GRAY2BGR);
    static int num = 0;

    for (int n = 0; n < nCheckerboardsEach; n++) {
        for (int i = 0; i < patternSizes.size(); i++) {
            cv::Size patternSize = patternSizes[i];
            std::vector<cv::Point2f> corners;

            cv::findChessboardCorners(image_copy, patternSize, corners);
            if (corners.size() == 0) {  // pattern not found
                std::cout << "pattern (" << patternSize.width << ", " << patternSize.height << ") not found" << std::endl;

                corners.assign(patternSize.area(), cv::Point2f(-1, -1));
            }
            else {  // pattern found
                cv::cornerSubPix(image, corners, cv::Size(4, 4), cv::Size(-1,-1), criteria);

                bool found;
                cv::drawChessboardCorners(draw_image, patternSize, corners, found);

                // hide detected region
                cv::Point2f top_left = corners[0] - cv::Point2f(20, 20);
                cv::Point2f bottom_right = corners[patternSize.area() - 1] + cv::Point2f(20, 20);
                cv::rectangle(image_copy, cv::Rect(top_left, bottom_right), cv::Scalar(0, 0, 0), cv::FILLED);
            }

            total_corners[i] = corners;
        }
    }


    cv::imwrite("output_logs/landmarks/frame" + std::to_string(num) + "_checkboard_corners.png", draw_image);
    num++;

    return total_corners;
}


std::vector<cv::Point2f> Tester::readKeypoints(const std::string dir_name, const int frame_id, const int frame_offset) {
    std::ifstream kp_file;

    int kp_frame_idx = frame_id + frame_offset;
    std::string kp_filename = dir_name + "keypoints/frame" + std::to_string(kp_frame_idx) + ".txt";
    kp_file.open(kp_filename);

    std::vector<cv::Point2f> result;

    // Keypoints
    if (!kp_file.is_open()) {
        std::cout << "Can not open keypoint file. " << kp_filename << std::endl;
        return result;
    }

    std::string line;
    std::getline(kp_file, line);  // skip the first line

    while (std::getline(kp_file, line)) {
        std::stringstream ss(line);

        std::string word;
        std::vector<std::string> words;

        while (getline(ss, word, ' ')) {
            words.push_back(word);
        }

        cv::KeyPoint kp;
        for (int i = 0; i < words.size(); i++) {
            std::string word = words[i];

            switch(i) {
                case 0:
                    kp.angle = std::stof(word);
                    break;
                case 1:
                    kp.class_id = std::stoi(word);
                    break;
                case 2:
                    kp.octave = std::stoi(word);
                    break;
                case 3:
                    word = word.substr(1, word.size()-2);
                    kp.pt.x = std::stof(word);
                    break;
                case 4:
                    word = word.substr(0, word.size()-1);
                    kp.pt.y = std::stof(word);
                    break;
                case 5:
                    kp.response = std::stof(word);
                    break;
                case 6:
                    kp.size = std::stof(word);
                    break;
                default:
                    std::cout << "keypoint reading error" << std::endl;
                    return result;
            }
        }
        result.push_back(kp.pt);
    }
    kp_file.close();

    std::cout << "successfully read keypoint file." << kp_filename << std::endl;

    return result;
}

std::vector<TestMatch> Tester::readMatches(const std::string dir_name, const int prevFrame_id, const int frame_offset) {
    std::ifstream match_file;

    int match_file_idx = prevFrame_id + frame_offset;
    std::string match_filename = dir_name + "matches/match_" + std::to_string(match_file_idx) + "_" + std::to_string(match_file_idx+1) + ".txt";
    match_file.open(match_filename);

    std::vector<TestMatch> result;

    // Matches
    if (!match_file.is_open()) {
        std::cout << "Can not open match file. " << match_filename << std::endl;
        return result;
    }

    std::string line;
    std::getline(match_file, line);  // skip the first line

    while (std::getline(match_file, line)) {
        std::stringstream ss(line);

        std::string word;
        std::vector<int> match_elem;
        while (std::getline(ss, word, ' ')) {
            match_elem.push_back(std::stoi(word));
        }

        if (match_elem[1] != -1) {
            TestMatch match(match_elem[0], match_elem[1]);
            match.distance = match_elem[2];

            result.push_back(match);
        }
    }
    match_file.close();

    std::cout << "successfully read match file." << match_filename << std::endl;

    return result;
}


void Tester::filterMatches(const std::shared_ptr<Frame> &pFrame, std::vector<TestMatch> &matches, const std::shared_ptr<Configuration> &pConfig) {
    cv::Size patch_size = cv::Size(pConfig->patch_width_, pConfig->patch_height_);
    cv::Size image_size = cv::Size(pFrame->image_.cols, pFrame->image_.rows);

    int row_iter = (image_size.height - 1) / patch_size.height;
    int col_iter = (image_size.width - 1) / patch_size.width;

    std::vector<std::vector<int>> bins(row_iter + 1, std::vector<int>(col_iter + 1));
    std::cout << "bin size: (" << bins.size() << ", " << bins[0].size() << ")" << std::endl;

    std::vector<TestMatch> filtered_matches;

    // sort matches
    std::sort(matches.begin(), matches.end(), [](const TestMatch& a, const TestMatch& b) {
        return a.distance < b.distance;
    });

    // filtering
    for (int i = 0; i < matches.size(); i++) {
        cv::Point2f kp_pt = pFrame->keypoints_pt_[matches[i].queryIdx];

        int bin_cnt = bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width];
        if (bin_cnt < pConfig->kps_per_patch_) {
            filtered_matches.push_back(matches[i]);

            bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width]++;
        }
    }

    matches = filtered_matches;
}



