#include "utils.hpp"

Utils::Utils(std::shared_ptr<Configuration> pConfig) {
    pConfig_ = pConfig;
}

void Utils::drawFramesLandmarks(const std::vector<std::shared_ptr<Frame>> &frames) {
    for (const auto pFrame : frames) {
        cv::Mat frame_img;
        cv::cvtColor(pFrame->image_, frame_img, cv::COLOR_GRAY2BGR);

        for (int i = 0; i < pFrame->landmarks_.size(); i++) {
            int keypoint_idx = pFrame->landmarks_[i]->observations_.find(pFrame->id_)->second;
            cv::Point2f kp_pt = pFrame->keypoints_[keypoint_idx].pt;

            cv::rectangle(frame_img,
                        kp_pt - cv::Point2f(5, 5),
                        kp_pt + cv::Point2f(5, 5),
                        cv::Scalar(0, 255, 0));  // green
        }
        cv::putText(frame_img, "frame" + std::to_string(pFrame->frame_image_idx_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

        cv::imwrite("output_logs/intra_frames/frame" + std::to_string(pFrame->frame_image_idx_) + "_landmarks.png", frame_img);
    }
}

void Utils::drawReprojectedLandmarks(const std::vector<std::shared_ptr<Frame>> &frames) {
    for (const auto pFrame: frames) {
        cv::Mat frame_img;
        cv::cvtColor(pFrame->image_, frame_img, cv::COLOR_GRAY2BGR);

        Eigen::Matrix3d K;
        cv::cv2eigen(pFrame->pCamera_->intrinsic_, K);

        Eigen::Isometry3d w_T_c = pFrame->pose_;
        Eigen::Isometry3d c_T_w = w_T_c.inverse();

        for (const auto pLandmark : pFrame->landmarks_) {
            Eigen::Vector3d landmark_point_3d = pLandmark->point_3d_;
            Eigen::Vector4d landmark_point_3d_homo(landmark_point_3d[0],
                                                    landmark_point_3d[1],
                                                    landmark_point_3d[2],
                                                    1);

            Eigen::Vector3d projected_point_homo = K * c_T_w.matrix().block<3, 4>(0, 0) * landmark_point_3d_homo;

            cv::Point2f projected_point(projected_point_homo[0] / projected_point_homo[2],
                                        projected_point_homo[1] / projected_point_homo[2]);
            cv::Point2f measurement_point = pFrame->keypoints_[pLandmark->observations_.find(pFrame->id_)->second].pt;

            cv::rectangle(frame_img,
                    measurement_point - cv::Point2f(5, 5),
                    measurement_point + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green
            cv::circle(frame_img, projected_point, 2, cv::Scalar(0, 0, 255));
            cv::line(frame_img, measurement_point, projected_point, cv::Scalar(0, 0, 255));
        }
        cv::putText(frame_img, "frame" + std::to_string(pFrame->frame_image_idx_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

        cv::imwrite("output_logs/reprojected_landmarks/frame" + std::to_string(pFrame->frame_image_idx_) + "_proj.png", frame_img);
    }
}

void Utils::drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<cv::DMatch> &good_matches,
                                    const cv::Mat &mask,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pFrame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, good_matches, cv::Mat(), triangulated_kps, prev_projected_pts, curr_projected_pts);

    int inlier_cnt = 0;
    for (int i = 0; i < good_matches.size(); i++) {
        cv::Point2f measurement_point0 = pPrev_frame->keypoints_[good_matches[i].queryIdx].pt;
        cv::Point2f measurement_point1 = pFrame->keypoints_[good_matches[i].trainIdx].pt;

        // draw images
        cv::rectangle(prev_frame_img,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255));  // green, (yellow)
        cv::circle(prev_frame_img,
                    prev_projected_pts[i],
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(prev_frame_img,
                    measurement_point0,
                    prev_projected_pts[i],
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::rectangle(curr_frame_img,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255));  // green, (yellow)
        cv::circle(curr_frame_img,
                    curr_projected_pts[i],
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(curr_frame_img,
                    measurement_point1,
                    curr_projected_pts[i],
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)

        if (mask.at<unsigned char>(i) == 1) {
            inlier_cnt++;
        }
    }
    cv::Mat result_image;
    cv::hconcat(prev_frame_img, curr_frame_img, result_image);

    cv::putText(result_image, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pFrame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(result_image, "#matched landmarks: " + std::to_string(good_matches.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(result_image, "#inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/reprojected_landmarks/frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + "frame" + std::to_string(pFrame->frame_image_idx_) + "_proj.png", result_image);
}

void Utils::drawKeypoints(std::shared_ptr<Frame> pFrame,
                        std::string folder,
                        std::string tail) {
    int id = pFrame->frame_image_idx_;
    cv::Mat image = pFrame->image_;
    std::vector<cv::KeyPoint> img_kps = pFrame->keypoints_;
    cv::Mat image_copy;

    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image.copyTo(image_copy);
    }

    for (int i = 0; i < img_kps.size(); i++) {
        cv::Point2f img_kp_pt = img_kps[i].pt;
        // draw images
        cv::rectangle(image_copy,
                    img_kp_pt - cv::Point2f(5, 5),
                    img_kp_pt + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (yellow)
        cv::circle(image_copy,
                    img_kp_pt,
                    1,
                    cv::Scalar(0, 0, 255));  // red, (blue)
    }
    cv::putText(image_copy, "frame" + std::to_string(id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image_copy, "#features: " + std::to_string(img_kps.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite(folder + "/frame" + std::to_string(id) + "_kps" + tail + ".png", image_copy);
}


void Utils::drawGrid(cv::Mat &image) {
    cv::Size patch_size = cv::Size(pConfig_->patch_width_, pConfig_->patch_height_);

    // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    int row_iter = (image.rows) / patch_size.height;
    int col_iter = (image.cols) / patch_size.width;

    // 가로선
    for (int i = 0; i < row_iter; i++) {
        int row = (i + 1) * patch_size.height;
        cv::line(image, cv::Point2f(0, row), cv::Point2f(image.cols, row), cv::Scalar(0, 0, 0), 2);
    }
    // 세로선
    for (int j = 0; j < col_iter; j++) {
        int col = (j + 1) * patch_size.width;
        cv::line(image, cv::Point2f(col, 0), cv::Point2f(col, image.rows), cv::Scalar(0, 0, 0), 2);
    }

    cv::putText(image, "patch width: " + std::to_string(patch_size.width),
                                    cv::Point(image.cols - 200, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image, "patch height: " + std::to_string(patch_size.height),
                                    cv::Point(image.cols - 200, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image, "keypoints / patch: " + std::to_string(pConfig_->kps_per_patch_),
                                    cv::Point(image.cols - 250, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
}

void Utils::alignPoses(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, std::vector<Eigen::Isometry3d> &aligned_est_poses) {
    double gt_length = 0, est_length = 0;

    std::vector<Eigen::Isometry3d> est_rel_poses;
    Eigen::Isometry3d prev_gt_pose = gt_poses[0];
    Eigen::Isometry3d prev_est_pose = est_poses[0];
    for (int i = 1; i < gt_poses.size(); i++) {
        Eigen::Isometry3d curr_gt_pose = gt_poses[i];
        Eigen::Isometry3d curr_est_pose = est_poses[i];

        Eigen::Isometry3d gt_rel_pose = prev_gt_pose.inverse() * curr_gt_pose;
        Eigen::Isometry3d est_rel_pose = prev_est_pose.inverse() * curr_est_pose;
        est_rel_poses.push_back(est_rel_pose);

        gt_length += gt_rel_pose.translation().norm();
        est_length += est_rel_pose.translation().norm();

        prev_gt_pose = curr_gt_pose;
        prev_est_pose = curr_est_pose;
    }

    double scale = gt_length / est_length;

    aligned_est_poses.push_back(Eigen::Isometry3d::Identity());
    for (int j = 0; j < est_rel_poses.size(); j++) {
        Eigen::Isometry3d aligned_est_rel_pose(est_rel_poses[j]);
        aligned_est_rel_pose.translation() = est_rel_poses[j].translation() * scale;

        aligned_est_poses.push_back(aligned_est_poses[j] * aligned_est_rel_pose);
    }
}

std::vector<Eigen::Isometry3d> Utils::calcRPE(const std::vector<std::shared_ptr<Frame>> &frames) {
    std::vector<Eigen::Isometry3d> rpe_vec;

    std::vector<Eigen::Isometry3d> gt_poses;
    loadGT(gt_poses);

    Eigen::Isometry3d est_rel_pose, gt_rel_pose;

    Eigen::Isometry3d prev_gt_pose = gt_poses[0];
    std::shared_ptr<Frame> pPrev_frame = frames[0];

    for (int i = 1; i < frames.size(); i++) {
        // GT relative pose
        Eigen::Isometry3d curr_gt_pose = gt_poses[i];
        gt_rel_pose = prev_gt_pose.inverse() * curr_gt_pose;

        // estimated relative pose
        std::shared_ptr<Frame> pCurr_frame = frames[i];
        est_rel_pose = pPrev_frame->pose_.inverse() * pCurr_frame->pose_;

        // calculate the relative pose error
        Eigen::Isometry3d relative_pose_error = gt_rel_pose.inverse() * est_rel_pose;
        rpe_vec.push_back(relative_pose_error);

        prev_gt_pose = curr_gt_pose;
    }

    return rpe_vec;
}

std::vector<Eigen::Isometry3d> Utils::calcRPE(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses) {
    std::vector<Eigen::Isometry3d> rpe_vec;

    Eigen::Isometry3d est_rel_pose, gt_rel_pose;

    Eigen::Isometry3d prev_gt_pose = gt_poses[0];
    Eigen::Isometry3d prev_est_pose = est_poses[0];
    for (int i = 1; i < gt_poses.size(); i++) {
        // GT relative pose
        Eigen::Isometry3d curr_gt_pose = gt_poses[i];
        Eigen::Isometry3d curr_est_pose = est_poses[i];

        gt_rel_pose = prev_gt_pose.inverse() * curr_gt_pose;
        est_rel_pose = prev_est_pose.inverse() * curr_est_pose;

        // calculate the relative pose error
        Eigen::Isometry3d relative_pose_error = gt_rel_pose.inverse() * est_rel_pose;
        rpe_vec.push_back(relative_pose_error);

        prev_gt_pose = curr_gt_pose;
    }

    return rpe_vec;
}

void Utils::calcRPE_rt(const std::vector<std::shared_ptr<Frame>> &frames, double &_rpe_rot, double &_rpe_trans) {
    std::vector<Eigen::Isometry3d> rpe_vec = calcRPE(frames);

    int num_rpe = rpe_vec.size();
    double acc_trans_error = 0;
    double acc_theta = 0;

    for (int i = 0; i < rpe_vec.size(); i++) {
        Eigen::Isometry3d rpe = rpe_vec[i];

        // RPE rotation
        Eigen::Matrix3d rotation = rpe.rotation();
        double rotation_trace = rotation.trace();
        double theta = acos((rotation_trace - 1) / 2);
        acc_theta += theta;

        // RPE translation
        double translation_error = rpe.translation().norm() * rpe.translation().norm();
        acc_trans_error += translation_error;
    }

    // mean RPE
    _rpe_rot = acc_theta / double(num_rpe);
    _rpe_trans = sqrt(acc_trans_error / double(num_rpe));
}

void Utils::calcRPE_rt(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, double &_rpe_rot, double &_rpe_trans) {
    std::vector<Eigen::Isometry3d> rpe_vec = calcRPE(gt_poses, est_poses);

    int num_rpe = rpe_vec.size();
    double acc_trans_error = 0;
    double acc_theta = 0;

    for (int i = 0; i < rpe_vec.size(); i++) {
        Eigen::Isometry3d rpe = rpe_vec[i];

        // RPE rotation
        Eigen::Matrix3d rotation = rpe.rotation();
        double rotation_trace = rotation.trace();
        double theta = acos((rotation_trace - 1) / 2);
        acc_theta += theta;

        // RPE translation
        double translation_error = rpe.translation().norm() * rpe.translation().norm();
        acc_trans_error += translation_error;
    }

    // mean RPE
    _rpe_rot = acc_theta / double(num_rpe);
    _rpe_trans = sqrt(acc_trans_error / double(num_rpe));
}

void Utils::loadGT(std::vector<Eigen::Isometry3d> &_gt_poses) {
    std::ifstream gt_poses_file(pConfig_->gt_path_);
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    std::vector<Eigen::Isometry3d> gt_poses;

    for (int l = 0; l < pConfig_->frame_offset_; l++) {
        std::getline(gt_poses_file, line);
    }

    for (int i = 0; i < pConfig_->num_frames_; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);
        if (pConfig_->is_kitti_) {  // KITTI format
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

        Eigen::Matrix3d rotation_mat;
        rotation_mat << r11, r12, r13,
                        r21, r22, r23,
                        r31, r32, r33;
        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;
        Eigen::Isometry3d gt_pose;
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;
        gt_poses.push_back(gt_pose);
    }

    _gt_poses = gt_poses;
}

Eigen::Isometry3d Utils::getGT(int &frame_idx){
    std::ifstream gt_poses_file(pConfig_->gt_path_);
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    std::vector<Eigen::Isometry3d> gt_poses;

    // skip not interesting lines
    for (int l = 0; l < pConfig_->frame_offset_ + frame_idx; l++) {
        std::getline(gt_poses_file, line);
    }

    // read the line
    std::getline(gt_poses_file, line);
    std::stringstream ssline(line);
    if (pConfig_->is_kitti_) {  // KITTI format
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

    // rotation
    Eigen::Matrix3d rotation_mat;
    rotation_mat << r11, r12, r13,
                    r21, r22, r23,
                    r31, r32, r33;

    // translation
    Eigen::Vector3d translation_mat;
    translation_mat << t1, t2, t3;

    // Transformation
    Eigen::Isometry3d gt_pose;
    gt_pose.linear() = rotation_mat;
    gt_pose.translation() = translation_mat;

    return gt_pose;
}

double Utils::calcReprojectionError(const std::vector<std::shared_ptr<Frame>> &frames) {
    double reprojection_error = 0;

    for (const auto pFrame: frames) {
        cv::Mat frame_img;
        cv::cvtColor(pFrame->image_, frame_img, cv::COLOR_GRAY2BGR);

        Eigen::Matrix3d K;
        cv::cv2eigen(pFrame->pCamera_->intrinsic_, K);

        Eigen::Isometry3d w_T_c = pFrame->pose_;
        Eigen::Isometry3d c_T_w = w_T_c.inverse();

        double error = 0;
        for (const auto pLandmark : pFrame->landmarks_) {
            Eigen::Vector3d landmark_point_3d = pLandmark->point_3d_;
            Eigen::Vector4d landmark_point_3d_homo(landmark_point_3d[0],
                                                    landmark_point_3d[1],
                                                    landmark_point_3d[2],
                                                    1);

            Eigen::Vector3d projected_point_homo = K * c_T_w.matrix().block<3, 4>(0, 0) * landmark_point_3d_homo;

            cv::Point2f projected_point(projected_point_homo[0] / projected_point_homo[2],
                                        projected_point_homo[1] / projected_point_homo[2]);
            cv::Point2f measurement_point = pFrame->keypoints_[pLandmark->observations_.find(pFrame->id_)->second].pt;

            cv::Point2f error_vector = projected_point - measurement_point;
            error = sqrt(error_vector.x * error_vector.x + error_vector.y * error_vector.y);
            reprojection_error += error;
        }
    }

    return reprojection_error;
}

double Utils::calcReprojectionError(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<cv::DMatch> &matches,
                                    const cv::Mat &mask,
                                    const std::vector<Eigen::Vector3d> &landmark_points_3d) {
    // assume matches.size() = landmark_points_3d.size()
    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    reprojectLandmarks(pFrame, matches, cv::Mat(), landmark_points_3d, prev_projected_pts, curr_projected_pts);

    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    double error_prev = 0, error_curr = 0;
    double reproj_error_prev = 0, reproj_error_curr = 0;
    int point_cnt = 0;
    for (int i = 0; i < matches.size(); i++) {
        if (mask.at<unsigned char>(i) == 1) {
            cv::Point2f measurement_point0 = pPrev_frame->keypoints_[matches[i].queryIdx].pt;
            cv::Point2f measurement_point1 = pFrame->keypoints_[matches[i].trainIdx].pt;

            cv::Point2f error_prev_pt = (prev_projected_pts[i] - measurement_point0);
            cv::Point2f error_curr_pt = (prev_projected_pts[i] - measurement_point0);

            error_prev = sqrt(error_prev_pt.x * error_prev_pt.x + error_prev_pt.y * error_prev_pt.y);
            error_curr = sqrt(error_curr_pt.x * error_curr_pt.x + error_curr_pt.y * error_curr_pt.y);

            reproj_error_prev += error_prev;
            reproj_error_curr += error_curr;

            point_cnt++;
        }
    }

    double reprojection_error = ((reproj_error_prev + reproj_error_curr) / point_cnt) / 2;
    return reprojection_error;
}

void Utils::drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int target_frame_id, const int dup_count) {
    std::shared_ptr<Frame> pTarget_frame = frames[target_frame_id];

    for (auto pLandmark : pTarget_frame->landmarks_) {
        if (pLandmark->observations_.size() > dup_count) {
            for (auto observation : pLandmark->observations_) {
                std::shared_ptr<Frame> pTargetFrame = frames[observation.first];

                // frame image
                cv::Mat frame_img;
                cv::cvtColor(pTargetFrame->image_, frame_img, cv::COLOR_GRAY2BGR);
                cv::Point2f keypoint_pt = pTargetFrame->keypoints_[pLandmark->observations_.find(pTargetFrame->frame_image_idx_)->second].pt;

                // camera pose
                Eigen::Isometry3d w_T_c = pTargetFrame->pose_;
                Eigen::Isometry3d c_T_w = w_T_c.inverse();

                cv::Mat rotation, translation;
                cv::eigen2cv(c_T_w.rotation(), rotation);
                cv::eigen2cv(Eigen::Vector3d(c_T_w.translation()), translation);

                // landmark 3d point
                cv::Point3f landmark_point_3d(pLandmark->point_3d_.x(), pLandmark->point_3d_.y(), pLandmark->point_3d_.z());
                std::vector<cv::Point2f> projected_pts;

                // project landmark to image coordinate
                cv::projectPoints(std::vector<cv::Point3f>{landmark_point_3d}, rotation, translation, pTargetFrame->pCamera_->intrinsic_, cv::Mat(), projected_pts);

                // draw markings
                cv::rectangle(frame_img,
                            keypoint_pt - cv::Point2f(5, 5),
                            keypoint_pt + cv::Point2f(5, 5),
                            cv::Scalar(0, 255, 0));  // green
                cv::circle(frame_img, projected_pts[0], 2, cv::Scalar(0, 0, 255));
                cv::line(frame_img, keypoint_pt, projected_pts[0], cv::Scalar(0, 0, 255));

                cv::putText(frame_img, "frame" + std::to_string(pTargetFrame->frame_image_idx_),
                                        cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

                cv::imwrite("output_logs/landmarks/landmark" + std::to_string(pLandmark->id_) + "_frame" + std::to_string(pTargetFrame->frame_image_idx_) + ".png", frame_img);
            }
        }

    }
}

void Utils::reprojectLandmarks(const std::shared_ptr<Frame> &pFrame,
                            const std::vector<cv::DMatch> &matches,
                            const cv::Mat &mask,
                            const std::vector<Eigen::Vector3d> &landmark_points_3d,
                            std::vector<cv::Point2f> &prev_projected_pts,
                            std::vector<cv::Point2f> &curr_projected_pts) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pFrame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);

    Eigen::Matrix3d K;
    cv::cv2eigen(pFrame->pCamera_->intrinsic_, K);

    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = K * prev_proj * pPrev_frame->pose_.inverse().matrix();
    curr_proj = K * curr_proj * pFrame->pose_.inverse().matrix();

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

void Utils::filterKeypoints(const cv::Size image_size, std::vector<cv::KeyPoint> &img_kps, cv::Mat &img_descriptor) {
    cv::Size patch_size = cv::Size(pConfig_->patch_width_, pConfig_->patch_height_);

    int row_iter = (image_size.height - 1) / patch_size.height;
    int col_iter = (image_size.width - 1) / patch_size.width;

    std::vector<std::vector<int>> bins(row_iter + 1, std::vector<int>(col_iter + 1));
    std::cout << "bin size: (" << bins.size() << ", " << bins[0].size() << ")" << std::endl;

    std::vector<cv::KeyPoint> filtered_kps;
    cv::Mat filtered_descriptors;
    std::vector<cv::Mat> filtered_descriptors_vec;

    for (int i = 0; i < img_kps.size(); i++) {
        cv::Point2f kp_pt = img_kps[i].pt;

        int bin_cnt = bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width];
        if (bin_cnt < pConfig_->kps_per_patch_) {
            filtered_kps.push_back(img_kps[i]);
            filtered_descriptors_vec.push_back(img_descriptor.row(i));

            bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width]++;
        }
    }

    for (auto desc : filtered_descriptors_vec) {
        filtered_descriptors.push_back(desc);
    }

    img_kps = filtered_kps;
    img_descriptor = filtered_descriptors;
}



