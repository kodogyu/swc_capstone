#include "tester.hpp"
#include "visual_odometry.hpp"

Tester::Tester() {
    manual_kp_frame0 = {cv::Point2f(89.0, 92.0), cv::Point2f(279.65207, 147.67671), cv::Point2f(440.3918, 203.43727), cv::Point2f(434.0, 161.0), cv::Point2f(486.80063, 190.52914), cv::Point2f(661.3766, 148.82133), cv::Point2f(709.0, 136.0), cv::Point2f(807.58307, 125.778046), cv::Point2f(916.523, 110.59789), cv::Point2f(907.6637, 294.75223)};
    manual_kp_frame1 = {cv::Point2f(76.435524, 92.402176), cv::Point2f(274.63477, 149.50064), cv::Point2f(439.3174, 206.34424), cv::Point2f(435.0, 163.0), cv::Point2f(487.71252, 192.5296), cv::Point2f(664.4662, 149.73691), cv::Point2f(714.8803, 135.63571), cv::Point2f(816.14087, 125.12716), cv::Point2f(928.7381, 109.08112), cv::Point2f(942.7255, 307.60168)};
    manual_kp_frame2 = {cv::Point2f(65, 90), cv::Point2f(270, 149), cv::Point2f(434, 208), cv::Point2f(436, 163), cv::Point2f(498, 191), cv::Point2f(668, 151), cv::Point2f(720, 136), cv::Point2f(827, 125), cv::Point2f(939, 108), cv::Point2f(977, 324)};
    manual_kp_frame3 = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106), cv::Point2f(1035, 346)};
    manual_kp_frame4 = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104),    cv::Point2f(653, 124)};
    manual_kps_vec = {manual_kp_frame0, manual_kp_frame1, manual_kp_frame2, manual_kp_frame3, manual_kp_frame4};

    manual_match_0_1 = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    manual_match_1_2 = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    manual_match_2_3 = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8), TestMatch(9, 9)};
    manual_match_3_4 = {TestMatch(0, 0), TestMatch(1, 1), TestMatch(2, 2), TestMatch(3, 3), TestMatch(4, 4), TestMatch(5, 5), TestMatch(6, 6), TestMatch(7, 7), TestMatch(8, 8)};
    manual_matches_vec = {manual_match_0_1, manual_match_1_2, manual_match_2_3, manual_match_3_4};
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

    // init depths_
    pFrame->depths_.reserve(kp_pts.size());

    // init keypoint_landmark_
    pFrame->keypoint_landmark_.assign(kp_pts.size(), std::pair(-1, -1));
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

            // hard matching
            if (pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].second != -1) {  // landmark already exists
                new_landmark = false;

                int landmark_idx = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx].first;
                auto pLandmark = pPrev_frame->landmarks_[landmark_idx];


                // TEST mode에서는 모든 feature 쌍이 올바르게 매칭되었다고 가정.
                landmark_accepted = true;
            }

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
                // add information to curr_frame
                std::pair<int, int> prev_frame_kp_lm = pPrev_frame->keypoint_landmark_[good_matches[i].queryIdx];
                int landmark_id = prev_frame_kp_lm.second;
                pCurr_frame->landmarks_.push_back(pPrev_frame->landmarks_[prev_frame_kp_lm.first]);
                pCurr_frame->keypoint_landmark_[good_matches[i].trainIdx] = std::pair(pCurr_frame->landmarks_.size(), landmark_id);
                pCurr_frame->keypoints_3d_[good_matches[i].trainIdx] = w_keypoint_3d;

                // add information to the landmark
                std::shared_ptr<Landmark> pLandmark = pPrev_frame->landmarks_[prev_frame_kp_lm.first];
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









