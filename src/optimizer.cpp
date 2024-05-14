#include "optimizer.hpp"

LocalOptimizer::LocalOptimizer() {}

void LocalOptimizer::optimizeFrames(std::vector<std::shared_ptr<Frame>> &frames, bool verbose) {
    // create a graph
    gtsam::NonlinearFactorGraph graph;

    // stereo camera calibration object
    gtsam::Cal3_S2::shared_ptr K(
        new gtsam::Cal3_S2(frames[0]->pCamera_->fx_,
                                frames[0]->pCamera_->fy_,
                                frames[0]->pCamera_->s_,
                                frames[0]->pCamera_->cx_,
                                frames[0]->pCamera_->cy_));

    // create initial values
    gtsam::Values initial_estimates;

    // 1. Add Values and Factors
    const auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);  // std of 1px.

    // insert values and factors of the frames
    std::vector<int> frame_pose_map;  // frame_pose_map[pose_idx] = frame_id
    std::vector<int> landmark_idx_id_map;  // landmark_map[landmark_idx] = landmark_id
    std::map<int, int> landmark_id_idx_map;  // landmark_map[landmark_id] = landmark_idx

    int landmark_idx = 0;
    int landmarks_cnt = 0;
    for (int frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
        std::shared_ptr<Frame> pFrame = frames[frame_idx];
        gtsam::Pose3 frame_pose = gtsam::Pose3(gtsam::Rot3(pFrame->pose_.rotation()), gtsam::Point3(pFrame->pose_.translation()));
        // insert initial value of the frame pose
        initial_estimates.insert(gtsam::Symbol('x', frame_idx), frame_pose);

        for (const auto pLandmark : pFrame->landmarks_) {
            // insert initial value of the landmark
            std::map<int, int>::iterator landmark_map_itr = landmark_id_idx_map.find(pLandmark->id_);
            if (landmark_map_itr == landmark_id_idx_map.end()) {  // new landmark
                landmark_idx = landmarks_cnt;

                initial_estimates.insert<gtsam::Point3>(gtsam::Symbol('l', landmark_idx), gtsam::Point3(pLandmark->point_3d_));
                landmark_idx_id_map.push_back(pLandmark->id_);
                landmark_id_idx_map[pLandmark->id_] = landmark_idx;

                landmarks_cnt++;
            }
            else {
                landmark_idx = landmark_map_itr->second;
            }
            // 2D measurement
            std::map<int, int>::iterator observation_itr = pLandmark->observations_.find(pFrame->id_);
            if (observation_itr != pLandmark->observations_.end()) {

                cv::Point2f measurement_cv = pFrame->keypoints_pt_.at(observation_itr->second);
                gtsam::Point2 measurement(measurement_cv.x, measurement_cv.y);
                // add measurement factor
                graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(measurement, measurement_noise,
                                                                                                                gtsam::Symbol('x', frame_idx), gtsam::Symbol('l', landmark_idx),
                                                                                                                K);
            }
        }
    }


    // 2. prior factors
    gtsam::Pose3 first_pose = gtsam::Pose3(gtsam::Rot3(frames[0]->pose_.rotation()), gtsam::Point3(frames[0]->pose_.translation()));
    // gtsam::Pose3 second_pose = gtsam::Pose3(gtsam::Rot3(frames[1]->pose_.rotation()), gtsam::Point3(frames[1]->pose_.translation()));
    // add a prior on the first pose
    const auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.3), gtsam::Vector3::Constant(0.1)).finished());  // std of 0.3m for x,y,z, 0.1rad for r,p,y
    graph.addPrior(gtsam::Symbol('x', 0), first_pose, pose_noise);
    // // add a prior on the second pose (for scale)
    // graph.addPrior(gtsam::Symbol('x', 1), second_pose, pose_noise);
    // add a prior on the first landmark (for scale)
    auto point_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    graph.addPrior(gtsam::Symbol('l', 0), frames[0]->landmarks_[0]->point_3d_, point_noise);

    // 3. Optimize
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimates);

    // start timer [optimization]
    const std::chrono::time_point<std::chrono::steady_clock> optimization_start = std::chrono::steady_clock::now();

    gtsam::Values result = optimizer.optimize();

    // end timer [optimization]
    const std::chrono::time_point<std::chrono::steady_clock> optimization_end = std::chrono::steady_clock::now();
    auto optimization_diff = optimization_end - optimization_start;
    std::cout << "GTSAM Optimization elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(optimization_diff).count() << "[ms]" << std::endl;

    std::cout << "initial error = " << graph.error(initial_estimates) << std::endl;
    std::cout << "final error = " << graph.error(result) << std::endl;


    // 4. Recover result pose
    for (int frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
        std::shared_ptr<Frame> pFrame = frames[frame_idx];
        pFrame->pose_ = result.at<gtsam::Pose3>(gtsam::Symbol('x', frame_idx)).matrix();

        if (pFrame->id_ > 0) {
            std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();
            pFrame->relative_pose_ = pPrev_frame->pose_.inverse() * pFrame->pose_;
        }

        // Recover Landmark point
        for (int j = 0; j < pFrame->landmarks_.size(); j++) {
            std::shared_ptr<Landmark> pLandmark = pFrame->landmarks_[j];
            std::map<int, int>::iterator landmark_map_itr = landmark_id_idx_map.find(pLandmark->id_);
            if (landmark_map_itr != landmark_id_idx_map.end()) {
                pLandmark->point_3d_ = result.at<gtsam::Point3>(gtsam::Symbol('l', landmark_map_itr->second));
            }
        }
    }
    if (verbose) {
        graph.print("graph print:\n");
        result.print("optimization result:\n");
    }
}

// void LocalOptimizer::optimizeFramesCustom(std::vector<std::shared_ptr<Frame>> &frames, bool verbose) {
//     Eigen::Matrix<double, 6, 6> hessian;
//     Eigen::Vector2d b;
//     Eigen::Isometry3d initial_camera_pose = frames[0]->pose_;

//     calcHb(frames[0], hessian, b);

//     Eigen::Isometry3d optimized_camera_pose = hessian.ldlt().solve(-b) * initial_camera_pose;
// }

// void LocalOptimizer::calcHb(std::shared_ptr<Frame> pFrame, Eigen::Matrix<double, 6, 6> &hessian, Eigen::Vector2d &b) {

//     Eigen::Matrix<double, 6, 6> hessian_temp = Eigen::Matrix<double, 6, 6>::Zero();
//     Eigen::Vector2d b_temp = Eigen::Vector2d::Zero();

//     Eigen::Matrix3d K;
//     cv::cv2eigen(pFrame->pCamera_->intrinsic_, K);

//     for (auto pLandmark : pFrame->landmarks_) {
//         Eigen::Matrix<double, 2, 9> jacobian;

//         Eigen::Matrix<double, 2, 6> jacobian_camera;
//         Eigen::Matrix<double, 2, 3> jacobian_landmark;

//         Eigen::Matrix<double, 2, 3> jacobian_homo;
//         Eigen::Matrix<double, 3, 6> jacobian_projection;

//         Eigen::Isometry3d w_T_c = pFrame->pose_;
//         Eigen::Isometry3d c_T_w = w_T_c.inverse();

//         Eigen::Vector3d landmark_point_3d = pLandmark->point_3d_;
//         Eigen::Vector4d landmark_point_3d_homo(landmark_point_3d[0],
//                                                 landmark_point_3d[1],
//                                                 landmark_point_3d[2],
//                                                 1);

//         Eigen::Vector3d projected_point_homo = K * c_T_w.matrix().block<3, 4>(0, 0) * landmark_point_3d_homo;

//         jacobian_homo << 1 / projected_point_homo.z(), 0, -projected_point_homo.x() / (projected_point_homo.z() * projected_point_homo.z()),
//                         0, 1 / projected_point_homo.z(), -projected_point_homo.y() / (projected_point_homo.z() * projected_point_homo.z());
//         jacobian_projection.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
//         jacobian_projection.block<3, 3>(0, 3) << 0, projected_point_homo.z(), -projected_point_homo.y(),
//                                                 -projected_point_homo.z(), 0, projected_point_homo.x(),
//                                                 projected_point_homo.y(), -projected_point_homo.x(), 0;

//         jacobian_camera = jacobian_homo * K * jacobian_projection;
//         jacobian_landmark = jacobian_homo * K * c_T_w.linear();
//         jacobian.block<2, 6>(0, 0) = jacobian_camera;
//         jacobian.block<2, 3>(0, 6) = jacobian_landmark;

//         cv::Point2f projected_point(projected_point_homo[0] / projected_point_homo[2],
//                                         projected_point_homo[1] / projected_point_homo[2]);
//         cv::Point2f measurement_point = pFrame->keypoints_[pLandmark->observations_.find(pFrame->id_)->second].pt;

//         cv::Point2f error_vector = projected_point - measurement_point;
//         Eigen::Vector2d error_vec(error_vec[0], error_vec[1]);

//         hessian_temp += jacobian_camera.transpose() * jacobian_camera;
//         b_temp += jacobian_camera.transpose() * error_vec;
//     }

//     hessian = hessian_temp;
//     b = b_temp;
// }

