#include "logger.hpp"

Logger::Logger() {
    trajectory_file_path_ = "output_logs/trajectory.csv";
    keypoint_file_path_ = "output_logs/keypoints.csv";
    landmark_file_path_ = "output_logs/landmarks.csv";
    cost_file_path_ = "output_logs/time_cost.csv";
    scale_file_path_ = "output_logs/scales.csv";
    rpe_file_path_ = "output_logs/rpe.csv";
}

void Logger::logTrajectory(std::vector<Eigen::Isometry3d> poses) const {
    std::ofstream trajectory_file(trajectory_file_path_);
    trajectory_file << "qw,qx,qy,qz,x,y,z\n";
    for (auto pose : poses) {
        Eigen::Quaterniond quaternion(pose.rotation());
        Eigen::Vector3d position = pose.translation();
        trajectory_file << quaternion.w() << "," << quaternion.x() << "," << quaternion.y() << "," << quaternion.z() << ","
                    << position.x() << "," << position.y() << "," << position.z() << "\n";
    }
}

void Logger::logKeypoints(std::vector<cv::Mat> keypoints_3d_vec) const {
    std::ofstream keypoints_file(keypoint_file_path_);
    for (int i = 0; i < keypoints_3d_vec.size(); i++) {
        cv::Mat keypoints = keypoints_3d_vec[i];
        keypoints_file << "# " << i << "\n";
        for (int j = 0; j < keypoints.cols; j++) {
            keypoints_file << keypoints.at<double>(0,j) << "," << keypoints.at<double>(1,j) << "," << keypoints.at<double>(2,j) << "\n";
        }
    }
}

void Logger::logLandmarks(std::vector<std::shared_ptr<Frame>> frames) const {
    std::ofstream landmarks_file(landmark_file_path_);
    for (auto pFrame : frames) {
        landmarks_file << "# " << pFrame->id_ << "\n";
        for (auto pLandmark : pFrame->landmarks_) {
            Eigen::Vector3d landmark_point_3d = pLandmark->point_3d_;
            landmarks_file << landmark_point_3d[0] << "," << landmark_point_3d[1] << "," << landmark_point_3d[2] << "\n";
        }
    }
}

void Logger::logTimecosts(std::vector<int64_t> feature_extraction_costs,
                        std::vector<int64_t> feature_matching_costs,
                        std::vector<int64_t> motion_estimation_costs,
                        std::vector<int64_t> triangulation_costs,
                        std::vector<int64_t> scaling_costs,
                        std::vector<int64_t> total_time_costs) const {
    std::ofstream cost_file(cost_file_path_);
    cost_file << "feature extraction(us),feature matching(us),motion estimation(us),triangulation(us),scaling(us),total time(us)\n";
    for (int i = 0; i < feature_extraction_costs.size(); i++) {
        cost_file << feature_extraction_costs[i] << "," << feature_matching_costs[i] << "," << motion_estimation_costs[i] << ","
                    << triangulation_costs[i] << "," << scaling_costs[i] << "," << total_time_costs[i] << "\n";
    }

}

void Logger::logScales(std::vector<double> scales, std::vector<double> gt_scales) const {
    std::ofstream scale_file(scale_file_path_);
    scale_file << "estimated scale,GT scale\n";
    for (int i = 0; i < scales.size(); i++) {
        scale_file << scales[i] << "," << gt_scales[i] << "\n";
    }
}

void Logger::logRPE(double rpe_rot, double rpe_trans) const {
    std::ofstream rpe_file(rpe_file_path_);
    rpe_file << "RPEr,RPEt\n";
    rpe_file << rpe_rot << "," << rpe_trans << "\n";
}
