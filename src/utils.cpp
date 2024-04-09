#include "utils.hpp"

Utils::Utils() {}

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
        cv::putText(frame_img, "frame" + std::to_string(pFrame->id_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

        cv::imwrite("output_logs/intra_frames/frame" + std::to_string(pFrame->id_) + "_landmarks.png", frame_img);
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
        cv::putText(frame_img, "frame" + std::to_string(pFrame->id_),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

        cv::imwrite("output_logs/reprojected_landmarks/frame" + std::to_string(pFrame->id_) + "_proj.png", frame_img);
    }
}

std::vector<Eigen::Isometry3d> Utils::calcRPE(const std::string gt_path, const std::vector<std::shared_ptr<Frame>> &frames) {
    std::vector<Eigen::Isometry3d> rpe_vec;

    std::vector<Eigen::Isometry3d> gt_poses;
    loadGT(gt_path, gt_poses);

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
    }

    return rpe_vec;
}

void Utils::calcRPE_rt(const std::string gt_path, const std::vector<std::shared_ptr<Frame>> &frames, double &_rpe_rot, double &_rpe_trans) {
    std::vector<Eigen::Isometry3d> rpe_vec = calcRPE(gt_path, frames);

    int num_frames = frames.size();
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
    _rpe_rot = acc_theta / double(num_frames);
    _rpe_trans = sqrt(acc_trans_error / double(num_frames));
}

void Utils::loadGT(std::string gt_path, std::vector<Eigen::Isometry3d> &_gt_poses) {
    std::ifstream gt_poses_file(gt_path);
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    std::vector<Eigen::Isometry3d> gt_poses;

    while(std::getline(gt_poses_file, line)) {
        std::stringstream ssline(line);
        // if (is_kitti_) {  // KITTI format
        ssline
            >> r11 >> r12 >> r13 >> t1
            >> r21 >> r22 >> r23 >> t2
            >> r31 >> r32 >> r33 >> t3;
        // }
        // else {
        //     ssline >> no_frame
        //             >> r11 >> r12 >> r13 >> t1
        //             >> r21 >> r22 >> r23 >> t2
        //             >> r31 >> r32 >> r33 >> t3;
        // }

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

void Utils::drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int pivot_frame_idx, const int dup_count) {
    std::shared_ptr<Frame> pPivotFrame = frames[pivot_frame_idx];

    for (auto pLandmark : pPivotFrame->landmarks_) {
        if (pLandmark->observations_.size() > dup_count) {
            for (auto observation : pLandmark->observations_) {
                std::shared_ptr<Frame> pTargetFrame = frames[observation.first];

                cv::Mat frame_img;
                cv::cvtColor(pTargetFrame->image_, frame_img, cv::COLOR_GRAY2BGR);
                cv::Point2f keypoint_pt = pTargetFrame->keypoints_[pLandmark->observations_.find(pTargetFrame->id_)->second].pt;

                Eigen::Isometry3d w_T_c = pTargetFrame->pose_;
                Eigen::Isometry3d c_T_w = w_T_c.inverse();

                cv::Mat rotation, translation;
                cv::eigen2cv(c_T_w.rotation(), rotation);
                cv::eigen2cv(Eigen::Vector3d(c_T_w.translation()), translation);

                cv::Point3f landmark_point_3d(pLandmark->point_3d_.x(), pLandmark->point_3d_.y(), pLandmark->point_3d_.z());
                std::vector<cv::Point2f> projected_pts;

                cv::projectPoints(std::vector<cv::Point3f>{landmark_point_3d}, rotation, translation, pTargetFrame->pCamera_->intrinsic_, cv::Mat(), projected_pts);

                cv::rectangle(frame_img,
                            keypoint_pt - cv::Point2f(5, 5),
                            keypoint_pt + cv::Point2f(5, 5),
                            cv::Scalar(0, 255, 0));  // green
                cv::circle(frame_img, projected_pts[0], 2, cv::Scalar(0, 0, 255));
                cv::line(frame_img, keypoint_pt, projected_pts[0], cv::Scalar(0, 0, 255));

                cv::putText(frame_img, "frame" + std::to_string(pTargetFrame->id_),
                                        cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

                cv::imwrite("output_logs/landmarks/landmark" + std::to_string(pLandmark->id_) + "_frame" + std::to_string(pTargetFrame->id_) + ".png", frame_img);
            }
        }

    }
}

