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
            cv::Point2f kp_pt = pFrame->keypoints_pt_[keypoint_idx];

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

void Utils::drawKeypoints(const std::shared_ptr<Frame> &pFrame) {
    cv::Mat frame_img;
    cv::cvtColor(pFrame->image_, frame_img, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < pFrame->keypoints_pt_.size(); i++) {
        cv::Point2f kp_pt = pFrame->keypoints_pt_[i];

        cv::rectangle(frame_img,
                    kp_pt - cv::Point2f(5, 5),
                    kp_pt + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green

        cv::line(frame_img,
                kp_pt,
                kp_pt,
                cv::Scalar(0, 0, 255));  // red
    }
    cv::putText(frame_img, "frame" + std::to_string(pFrame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/landmarks/frame" + std::to_string(pFrame->frame_image_idx_) + "_keypoints.png", frame_img);
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
            cv::Point2f measurement_point = pFrame->keypoints_pt_[pLandmark->observations_.find(pFrame->id_)->second];

            cv::rectangle(frame_img,
                    measurement_point - cv::Point2f(5, 5),
                    measurement_point + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green
            cv::circle(frame_img, projected_point, 2, cv::Scalar(0, 0, 255));
            cv::line(frame_img, measurement_point, projected_point, cv::Scalar(0, 0, 255));
        }
        cv::putText(frame_img, "frame" + std::to_string(pFrame->frame_image_idx_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

        cv::imwrite("output_logs/intra_frames/reprojected_landmarks/frame" + std::to_string(pFrame->frame_image_idx_) + "_proj.png", frame_img);
    }
}

void Utils::drawReprojectedLandmarks(const std::shared_ptr<Frame> &pFrame,
                                    const std::vector<cv::DMatch> &good_matches,
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

void Utils::drawCvReprojectedLandmarks(const std::shared_ptr<Frame> &pPrev_frame,
                                    const std::vector<cv::Point2f> &image0_kp_pts,
                                    const std::shared_ptr<Frame> &pCurr_frame,
                                    const std::vector<cv::Point2f> &image1_kp_pts,
                                    const std::vector<Eigen::Vector3d> &triangulated_kps,
                                    const cv::Mat &pose_mask) {
    // reproject points
    cv::Mat prev_rvec, prev_tvec, curr_rvec, curr_tvec;
    Eigen::Matrix3d rvec_temp;
    Eigen::Vector3d tvec_temp;
    rvec_temp = pPrev_frame->pose_.inverse().rotation();
    tvec_temp = pPrev_frame->pose_.inverse().translation();
    cv::eigen2cv(rvec_temp, prev_rvec);
    cv::eigen2cv(tvec_temp, prev_tvec);

    rvec_temp = pCurr_frame->pose_.inverse().rotation();
    tvec_temp = pCurr_frame->pose_.inverse().translation();
    cv::eigen2cv(rvec_temp, curr_rvec);
    cv::eigen2cv(tvec_temp, curr_tvec);

    std::vector<cv::Point3f> triangulated_kps_cv;
    for (auto kp : triangulated_kps) {
        cv::Point3f kp_cv(kp.x(), kp.y(), kp.z());
        triangulated_kps_cv.push_back(kp_cv);
    }

    std::vector<cv::Point2f> prev_projected_pts, curr_projected_pts;
    cv::projectPoints(triangulated_kps_cv, prev_rvec, prev_tvec, pPrev_frame->pCamera_->intrinsic_, cv::Mat(), prev_projected_pts);
    cv::projectPoints(triangulated_kps_cv, curr_rvec, curr_tvec, pCurr_frame->pCamera_->intrinsic_, cv::Mat(), curr_projected_pts);

    // draw reprojected points
    cv::Mat curr_frame_img, prev_frame_img;
    cv::cvtColor(pPrev_frame->image_, prev_frame_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pCurr_frame->image_, curr_frame_img, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < triangulated_kps.size(); i++) {
        cv::Point2f measurement_point0 = image0_kp_pts[i];
        cv::Point2f measurement_point1 = image1_kp_pts[i];

        // draw original keypoint (prev_frame)
        cv::rectangle(prev_frame_img,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 127, 255));  // green, (orange)

        // draw reprojected keypoint (prev_frame)
        cv::circle(prev_frame_img,
                    prev_projected_pts[i],
                    2,
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)

        // draw original-reprojected line (prev_frame)
        cv::line(prev_frame_img,
                    measurement_point0,
                    prev_projected_pts[i],
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)

        // draw original keypoint (curr_frame)
        cv::rectangle(curr_frame_img,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 127, 255));  // green, (orange)

        // draw reprojected keypoint (curr_frame)
        cv::circle(curr_frame_img,
                    curr_projected_pts[i],
                    2,
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)

        // draw original-reprojected line (curr_frame)
        cv::line(curr_frame_img,
                    measurement_point1,
                    curr_projected_pts[i],
                    pose_mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
    }
    cv::Mat result_image;
    cv::vconcat(prev_frame_img, curr_frame_img, result_image);

    cv::putText(result_image, "frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + std::to_string(pCurr_frame->frame_image_idx_),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("output_logs/inter_frames/reprojected_landmarks/frame" + std::to_string(pPrev_frame->frame_image_idx_) + "&" + "frame" + std::to_string(pCurr_frame->frame_image_idx_) + "_proj.png", result_image);
}

void Utils::drawKeypoints(std::shared_ptr<Frame> pFrame,
                        std::string folder,
                        std::string tail) {
    int id = pFrame->frame_image_idx_;
    cv::Mat image = pFrame->image_;
    // std::vector<cv::KeyPoint> img_kps = pFrame->keypoints_;
    std::vector<cv::Point2f> img_kps_pt = pFrame->keypoints_pt_;
    cv::Mat image_copy;

    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image.copyTo(image_copy);
    }

    // for (int i = 0; i < img_kps.size(); i++) {
    for (int i = 0; i < img_kps_pt.size(); i++) {
        cv::Point2f img_kp_pt = img_kps_pt[i];
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
    cv::putText(image_copy, "#features: " + std::to_string(img_kps_pt.size()),
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

void Utils::drawMatches(const std::shared_ptr<Frame> &pPrev_frame, const std::shared_ptr<Frame> &pCurr_frame, const std::vector<cv::DMatch> &good_matches) {
    cv::Mat image_matches;
    cv::drawMatches(pPrev_frame->image_, pPrev_frame->keypoints_,
                    pCurr_frame->image_, pCurr_frame->keypoints_,
                    good_matches, image_matches);
    cv::imwrite("output_logs/inter_frames/frame"
            + std::to_string(pPrev_frame->frame_image_idx_)
            + "&"
            + std::to_string(pCurr_frame->frame_image_idx_)
            + "_kp_matches(raw).png", image_matches);
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

Eigen::Isometry3d Utils::getGT(const int frame_idx){
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
            // cv::Point2f measurement_point = pFrame->keypoints_[pLandmark->observations_.find(pFrame->id_)->second].pt;
            cv::Point2f measurement_point = pFrame->keypoints_pt_[pLandmark->observations_.find(pFrame->id_)->second];

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

void Utils::drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int target_frame_id, const int dup_count) {
    std::shared_ptr<Frame> pTarget_frame = frames[target_frame_id];

    for (auto pLandmark : pTarget_frame->landmarks_) {
        if (pLandmark->observations_.size() > dup_count) {
            for (auto observation : pLandmark->observations_) {
                std::shared_ptr<Frame> pTargetFrame = frames[observation.first];

                // frame image
                cv::Mat frame_img;
                cv::cvtColor(pTargetFrame->image_, frame_img, cv::COLOR_GRAY2BGR);
                // cv::Point2f keypoint_pt = pTargetFrame->keypoints_[pLandmark->observations_.find(pTargetFrame->frame_image_idx_)->second].pt;
                cv::Point2f keypoint_pt = pTargetFrame->keypoints_pt_[pLandmark->observations_.find(pTargetFrame->frame_image_idx_)->second];

                // camera pose
                Eigen::Isometry3d w_T_c = pTargetFrame->pose_;
                Eigen::Isometry3d c_T_w = w_T_c.inverse();

                cv::Mat rotation, translation;
                Eigen::Matrix3d c_T_w_rot = c_T_w.rotation();
                cv::eigen2cv(c_T_w_rot, rotation);
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

void Utils::filterKeypoints(std::shared_ptr<Frame> &pFrame) {
    cv::Size patch_size = cv::Size(pConfig_->patch_width_, pConfig_->patch_height_);
    cv::Size image_size = cv::Size(pFrame->image_.cols, pFrame->image_.rows);

    int row_iter = (image_size.height - 1) / patch_size.height;
    int col_iter = (image_size.width - 1) / patch_size.width;

    // make bins
    std::vector<std::vector<int>> bins(row_iter + 1, std::vector<int>(col_iter + 1));
    std::cout << "bin size: (" << bins.size() << ", " << bins[0].size() << ")" << std::endl;

    std::vector<cv::KeyPoint> img_kps = pFrame->keypoints_;
    cv::Mat img_descriptor = pFrame->descriptors_;

    std::vector<cv::KeyPoint> filtered_kps;
    cv::Mat filtered_descriptors;
    std::vector<cv::Mat> filtered_descriptors_vec;
    // filtering
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

    pFrame->keypoints_ = filtered_kps;
    pFrame->descriptors_ = filtered_descriptors;
}

void Utils::filterMatches(std::shared_ptr<Frame> &pFrame, std::vector<cv::DMatch> &matches) {
    cv::Size patch_size = cv::Size(pConfig_->patch_width_, pConfig_->patch_height_);
    cv::Size image_size = cv::Size(pFrame->image_.cols, pFrame->image_.rows);

    int row_iter = (image_size.height - 1) / patch_size.height;
    int col_iter = (image_size.width - 1) / patch_size.width;

    std::vector<std::vector<int>> bins(row_iter + 1, std::vector<int>(col_iter + 1));
    std::cout << "bin size: (" << bins.size() << ", " << bins[0].size() << ")" << std::endl;

    std::vector<cv::DMatch> filtered_matches;

    // sort matches
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    // filtering
    for (int i = 0; i < matches.size(); i++) {
        cv::Point2f kp_pt = pFrame->keypoints_pt_[matches[i].queryIdx];

        int bin_cnt = bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width];
        if (bin_cnt < pConfig_->kps_per_patch_) {
            filtered_matches.push_back(matches[i]);

            bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width]++;
        }
    }

    matches = filtered_matches;
}

Eigen::Matrix3d Utils::findFundamentalMat(const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts) {
    // Normalized Eight-point algorithm

    // matrix W
    int num_keypoints = image0_kp_pts.size();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(num_keypoints, 9);
    // std::cout << "Matrix W:\n" << W.matrix() << std::endl;

    // concat points
    Eigen::MatrixXd P(3, num_keypoints), P_prime(3, num_keypoints);
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3d p(image0_kp_pts[i].x, image0_kp_pts[i].y, 1);
        Eigen::Vector3d p_prime(image1_kp_pts[i].x, image1_kp_pts[i].y, 1);

        P.block<3, 1>(0, i) = p;
        P_prime.block<3, 1>(0, i) = p_prime;
    }

    // mean, SSE
    Eigen::Vector3d P_mean = P.rowwise().mean();
    Eigen::Vector3d P_prime_mean = P_prime.rowwise().mean();
    // std::cout << "P_mean: \n" << P_mean << std::endl;
    // std::cout << "P_prime_mean: \n" << P_prime_mean << std::endl;

    Eigen::Vector3d P_sse = Eigen::Vector3d::Zero();
    Eigen::Vector3d P_prime_sse = Eigen::Vector3d::Zero();
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3d P_e = P.col(i) - P_mean;
        Eigen::Vector3d P_prime_e = P_prime.col(i) - P_prime_mean;

        P_sse += (P_e.array() * P_e.array()).matrix();
        P_prime_sse += (P_prime_e.array() * P_prime_e.array()).matrix();
    }
    // std::cout << "P_sse: \n" << P_sse << std::endl;
    // std::cout << "P_prime_sse: \n" << P_prime_sse << std::endl;

    // scale (P_sse.array() / num_keypoints = variance)
    Eigen::Vector3d P_scale = Eigen::sqrt(2*num_keypoints / P_sse.array());
    Eigen::Vector3d P_prime_scale = Eigen::sqrt(2*num_keypoints / P_prime_sse.array());
    // std::cout << "P_scale: \n" << P_scale << std::endl;
    // std::cout << "P_prime_scale: \n" << P_prime_scale << std::endl;

    // transform
    Eigen::Matrix3d P_transform;
    Eigen::Matrix3d P_prime_transform;
    P_transform << P_scale(0), 0, P_scale(0) * -P_mean(0),
                    0, P_scale(1), P_scale(1) * -P_mean(1),
                    0, 0, 1;
    P_prime_transform << P_prime_scale(0), 0, P_prime_scale(0) * -P_prime_mean(0),
                    0, P_prime_scale(1), P_prime_scale(1) * -P_prime_mean(1),
                    0, 0, 1;
    // std::cout << "P_transform: \n" << P_transform.matrix() << std::endl;
    // std::cout << "P_prime_transform: \n" << P_prime_transform.matrix() << std::endl;

    // Normalize keypoints
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3d p = P_transform * P.col(i);
        Eigen::Vector3d p_prime = P_prime_transform * P_prime.col(i);

        // pT * F * p' = 0
        Eigen::Matrix3d p_p_prime = p * p_prime.transpose();
        Eigen::Map<Eigen::RowVectorXd> p_p_prime_vec(p_p_prime.data(), p_p_prime.size());

        W.block<1, 9>(i, 0) = p_p_prime_vec;
        // std::cout << "pq[" << i << "] : " << p_p_prime_vec << std::endl;

        // // p'T * F * p = 0
        // Eigen::Matrix3d p_prime_p = p_prime * p.transpose();
        // Eigen::Map<Eigen::RowVectorXd> p_prime_p_vec(p_prime_p.data(), p_prime_p.size());

        // W.block<1, 9>(i, 0) = p_prime_p_vec;

    }
    // std::cout << "Matrix W:\n" << W.matrix() << std::endl;

    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> fundamental_vec = svd.matrixV().col(8);

    // rank 2
    // std::cout << "fundamental_vec: \n" << fundamental_vec.matrix() << std::endl;
    Eigen::Map<Eigen::MatrixXd> fundamental_mat_temp(fundamental_vec.data(), 3, 3);

    Eigen::JacobiSVD<Eigen::MatrixXd> f_svd(fundamental_mat_temp, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd fundamental_U = f_svd.matrixU();
    Eigen::MatrixXd fundamental_S = f_svd.singularValues().asDiagonal();
    Eigen::MatrixXd fundamental_V = f_svd.matrixV();

    // std::cout << "fundamental_S: \n" << fundamental_S.matrix() << std::endl;
    fundamental_S(2, 2) = 0;
    // std::cout << "fundamental_S: \n" << fundamental_S.matrix() << std::endl;

    // Fundamental Matrix
    Eigen::Matrix3d fundamental_mat = fundamental_U * fundamental_S * fundamental_V.transpose();
    // std::cout << "Fundamental Matrix: \n" << fundamental_mat.matrix() << std::endl;

    // Re-transform
    // fundamental_mat = P_transform.transpose() * fundamental_mat * P_prime_transform;
    fundamental_mat = P_prime_transform.transpose() * fundamental_mat * P_transform;
    fundamental_mat = fundamental_mat.array() / fundamental_mat(2, 2);
    // std::cout << "Fundamental Matrix: \n" << fundamental_mat.matrix() << std::endl;

    return fundamental_mat;
}

Eigen::Matrix3d Utils::findFundamentalMat(const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts,
                                            cv::Mat &mask, double inlier_prob, double ransac_threshold) {
    int sample_size = 8;
    Ransac ransac(sample_size, inlier_prob, ransac_threshold, image0_kp_pts, image1_kp_pts);

    ransac.run();

    mask = cv::Mat::zeros(image0_kp_pts.size(), 1, CV_8UC1);
    for (int i = 0; i < ransac.best_inlier_idxes_.size(); i++) {
        int idx = ransac.best_inlier_idxes_[i];
        mask.at<unsigned char>(idx) = 1;
    }

    return ransac.best_model_;
}

Eigen::Matrix3d Utils::findEssentialMat(const cv::Mat &intrinsic, const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts) {
    Eigen::Matrix3d fundamental_mat = findFundamentalMat(image0_kp_pts, image1_kp_pts);

    cv::Mat fundamental_mat_cv;
    cv::eigen2cv(fundamental_mat, fundamental_mat_cv);

    cv::Mat essential_mat_cv = intrinsic.t() * fundamental_mat_cv * intrinsic;
    // std::cout << "essential matrix:\n" << essential_mat_cv << std::endl;

    // SVD
    Eigen::Matrix3d essential_mat;
    cv::cv2eigen(essential_mat_cv, essential_mat);

    Eigen::JacobiSVD<Eigen::Matrix3d> e_svd(essential_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d essential_U = e_svd.matrixU();
    Eigen::Matrix3d essential_S = Eigen::Vector3d(1, 1, 0).asDiagonal();
    Eigen::Matrix3d essential_V = e_svd.matrixV();
    essential_mat = essential_U * essential_S * essential_V.transpose();
    std::cout << "essential matrix:\n" << essential_mat << std::endl;

    return essential_mat;
}

Eigen::Matrix3d Utils::findEssentialMat(const cv::Mat &intrinsic, const std::vector<cv::Point2f> &image0_kp_pts, const std::vector<cv::Point2f> &image1_kp_pts,
                                        cv::Mat &mask, double inlier_prob, double ransac_threshold) {
    Eigen::Matrix3d fundamental_mat = findFundamentalMat(image0_kp_pts, image1_kp_pts, mask, inlier_prob, ransac_threshold);

    cv::Mat fundamental_mat_cv;
    cv::eigen2cv(fundamental_mat, fundamental_mat_cv);

    cv::Mat essential_mat_cv = intrinsic.t() * fundamental_mat_cv * intrinsic;
    // std::cout << "essential matrix:\n" << essential_mat_cv << std::endl;

    // SVD
    Eigen::Matrix3d essential_mat;
    cv::cv2eigen(essential_mat_cv, essential_mat);

    Eigen::JacobiSVD<Eigen::Matrix3d> e_svd(essential_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d essential_U = e_svd.matrixU();
    Eigen::Matrix3d essential_S = Eigen::Vector3d(1, 1, 0).asDiagonal();
    Eigen::Matrix3d essential_V = e_svd.matrixV();
    essential_mat = essential_U * essential_S * essential_V.transpose();
    std::cout << "essential matrix:\n" << essential_mat << std::endl;

    Eigen::Vector3d essential_singular_values = e_svd.singularValues();

    return essential_mat;
}

void Utils::decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat U, S, VT;
    cv::SVD::compute(essential_mat, S, U, VT);

    // // OpenCV decomposeEssentialMat()
    // cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0,
    //                                         -1, 0, 0,
    //                                         0, 0, 1);

    // hand function.
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                                            1, 0, 0,
                                            0, 0, 1);


    // double det_U = cv::determinant(U);
    // double det_W = cv::determinant(W);
    // double det_VT = cv::determinant(VT);
    // std::cout << "det U: " << det_U << std::endl;
    // std::cout << "det W: " << det_W << std::endl;
    // std::cout << "det VT: " << det_VT << std::endl;

    R1 = U * W * VT;
    R2 = U * W.t() * VT;
    t = U.col(2);

    double det_R1 = cv::determinant(R1);
    double det_R2 = cv::determinant(R2);
    // std::cout << "det R1: " << det_R1 << std::endl;
    // std::cout << "det R2: " << det_R2 << std::endl;

    if (det_R1 < 0) {
        R1 = -R1;
    }
    if (det_R1 < 0) {
        R2 = -R2;
    }
    // std::cout << "R1:\n" << R1 << std::endl;
    // std::cout << "R2:\n" << R2 << std::endl;
    // std::cout << "t:\n" << t << std::endl;

    // std::vector<cv::Mat> pose_candidates(4);
    // cv::hconcat(std::vector<cv::Mat>{R1, t}, pose_candidates[0]);
    // cv::hconcat(std::vector<cv::Mat>{R1, -t}, pose_candidates[1]);
    // cv::hconcat(std::vector<cv::Mat>{R2, t}, pose_candidates[2]);
    // cv::hconcat(std::vector<cv::Mat>{R2, -t}, pose_candidates[3]);

    // for (int i = 0; i < pose_candidates.size(); i++) {
    //     std::cout << "pose_candidate [" << i << "]: \n" << pose_candidates[i] << std::endl;
    // }
}

int Utils::chiralityCheck(const cv::Mat &intrinsic,
                    const std::vector<cv::Point2f> &image0_kp_pts,
                    const std::vector<cv::Point2f> &image1_kp_pts,
                    const Eigen::Isometry3d &cam1_pose,
                    cv::Mat &mask) {
    // std::cout << "----- Utils::chiralityCheck -----" << std::endl;
    Eigen::Matrix3d intrinsic_eigen;
    cv::cv2eigen(intrinsic, intrinsic_eigen);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = intrinsic_eigen * prev_proj;
    // curr_proj = intrinsic_eigen * curr_proj * cam1_pose.matrix();
    curr_proj = intrinsic_eigen * curr_proj * cam1_pose.inverse().matrix();

    // std::cout << "cam1_pose:\n" << cam1_pose.matrix() << std::endl;
    // std::cout << "cam1_pose inverse:\n" << cam1_pose.inverse().matrix() << std::endl;

    int positive_cnt = 0;

    for (int i = 0; i < image0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = image0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = image0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = image1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = image1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector4d cam1_point_3d_homo = cam1_pose.inverse() * point_3d_homo;
        Eigen::Vector3d cam1_point_3d = cam1_point_3d_homo.head(3) / cam1_point_3d_homo[3];
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (point_3d.z() > 0 && cam1_point_3d.z() > 0 && point_3d.z() < 70) {
            mask.at<unsigned char>(i) = 1;
            positive_cnt++;
        }
        else {
            mask.at<unsigned char>(i) = 0;
        }
    }

    return positive_cnt;
}

void Utils::recoverPose(const cv::Mat &intrinsic,
                        const cv::Mat &essential_mat,
                        const std::vector<cv::Point2f> &image0_kp_pts,
                        const std::vector<cv::Point2f> &image1_kp_pts,
                        Eigen::Isometry3d &relative_pose,
                        cv::Mat &mask) {

    // Decompose essential matrix
    cv::Mat R1, R2, t;
    // cv::decomposeEssentialMat(essential_mat, R1, R2, t);
    // std::cout << "cv_R1:\n" << R1 << std::endl;
    // std::cout << "cv_R2:\n" << R2 << std::endl;
    // std::cout << "cv_t:\n" << t << std::endl;
    decomposeEssentialMat(essential_mat, R1, R2, t);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;

    std::vector<Eigen::Isometry3d> rel_pose_candidates(4, Eigen::Isometry3d::Identity());
    std::vector<cv::Mat> masks(4);

    for (int  i = 0; i < 4; i++) {
        masks[i] = mask.clone();
    }

    int valid_point_cnts[4];
    for (int k = 0; k < 4; k++) {
        if (k == 0) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 1) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        else if (k == 2) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 3) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        // rotation_mat << 0.999992,  0.00248744,  0.00303404,
        //                 -0.00249412,    0.999994,  0.00219979,
        //                 -0.00302855, -0.00220734,    0.999993;
        // translation_mat << 0.00325004, 0.00744242, -0.999967;

        // rel_pose_candidates[k].linear() = rotation_mat;
        // rel_pose_candidates[k].translation() = translation_mat;
        rel_pose_candidates[k].linear() = rotation_mat.transpose();
        rel_pose_candidates[k].translation() = - rotation_mat.transpose() * translation_mat;

        // std::cout << "pose candidate[" << k << "]:\n" << rel_pose_candidates[k].matrix() << std::endl;
        // std::cout << "R:\n" << rotation_mat.matrix() << std::endl;
        // std::cout << "t:\n" << translation_mat.matrix() << std::endl;

        valid_point_cnts[k] = chiralityCheck(intrinsic, image0_kp_pts, image1_kp_pts, rel_pose_candidates[k], masks[k]);
    }

    int max_cnt = 0, max_idx = 0;
    for (int k = 0; k < 4; k++) {
        std::cout << "cnt[" << k << "]: " << valid_point_cnts[k] << std::endl;
        if (valid_point_cnts[k] > max_cnt) {
            max_cnt = valid_point_cnts[k];
            max_idx = k;
        }
    }
    // std::cout << "max idx: " << max_idx << std::endl;
    relative_pose = rel_pose_candidates[max_idx];
    mask = masks[max_idx];
    std::cout << "mask count: " << countMask(mask) << std::endl;
    // std::cout << "relative pose:\n " << relative_pose.matrix() << std::endl;
}


int Utils::countMask(const cv::Mat &mask) {
    int count = 0;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            count++;
        }
    }
    return count;
}

void Utils::cv_triangulatePoints(const std::shared_ptr<Frame>& pPrev_frame, const std::vector<cv::Point2f> &prev_kp_pts,
                                const std::shared_ptr<Frame>& pCurr_frame, const std::vector<cv::Point2f> &curr_kp_pts,
                                const std::vector<cv::DMatch> &good_matches, std::vector<Eigen::Vector3d> &keypoints_3d) {
    cv::Mat points4D;
    cv::Mat prev_proj_mat, curr_proj_mat;
    cv::Mat prev_pose, curr_pose;
    Eigen::MatrixXd pose_temp;
    pose_temp = pPrev_frame->pose_.inverse().matrix().block<3, 4>(0, 0);
    cv::eigen2cv(pose_temp, prev_pose);
    pose_temp = pCurr_frame->pose_.inverse().matrix().block<3, 4>(0, 0);
    cv::eigen2cv(pose_temp, curr_pose);
    prev_proj_mat = pPrev_frame->pCamera_->intrinsic_ * prev_pose;
    curr_proj_mat = pCurr_frame->pCamera_->intrinsic_ * curr_pose;

    std::cout << "prev_proj_mat:\n" << prev_proj_mat << std::endl;
    std::cout << "curr_proj_mat:\n" << curr_proj_mat << std::endl;

    // triangulate
    cv::triangulatePoints(prev_proj_mat, curr_proj_mat, prev_kp_pts, curr_kp_pts, points4D);

    // homogeneous -> 3D
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat point_homo = points4D.col(i);
        Eigen::Vector3d point_3d(point_homo.at<float>(0) / point_homo.at<float>(3),
                                point_homo.at<float>(1) / point_homo.at<float>(3),
                                point_homo.at<float>(2) / point_homo.at<float>(3));

        keypoints_3d.push_back(point_3d);
    }

}

// triangulate all keypoints
void Utils::triangulateKeyPoints(std::shared_ptr<Frame> &pFrame,
                                        std::vector<cv::Point2f> img0_kp_pts,
                                        std::vector<cv::Point2f> img1_kp_pts,
                                        std::vector<Eigen::Vector3d> &triangulated_kps) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(pFrame->pCamera_->intrinsic_, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj * pPrev_frame->pose_.inverse().matrix();
    curr_proj = camera_intrinsic * curr_proj * pFrame->pose_.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        triangulated_kps.push_back(point_3d);
    }
}

// triangulate single keypoint
void Utils::triangulateKeyPoint(std::shared_ptr<Frame> &pFrame,
                                        cv::Point2f img0_kp_pt,
                                        cv::Point2f img1_kp_pt,
                                        Eigen::Vector3d &triangulated_kp) {
    std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();

    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(pFrame->pCamera_->intrinsic_, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj * pPrev_frame->pose_.inverse().matrix();
    curr_proj = camera_intrinsic * curr_proj * pFrame->pose_.inverse().matrix();

    Eigen::Matrix4d A;
    A.row(0) = img0_kp_pt.x * prev_proj.row(2) - prev_proj.row(0);
    A.row(1) = img0_kp_pt.y * prev_proj.row(2) - prev_proj.row(1);
    A.row(2) = img1_kp_pt.x * curr_proj.row(2) - curr_proj.row(0);
    A.row(3) = img1_kp_pt.y * curr_proj.row(2) - curr_proj.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
    Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
    triangulated_kp = point_3d;
}




