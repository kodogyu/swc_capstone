#include <fstream>

#include "common_includes.hpp"
#include "frame.hpp"

class Logger {
public:
    Logger();

    void logTrajectory(std::vector<Eigen::Isometry3d> poses) const;
    void logTrajectoryTxt(std::vector<Eigen::Isometry3d> poses) const;
    void logTrajectoryTxt(std::vector<std::shared_ptr<Frame>> frames) const;
    void logTrajectoryTxtAppend(Eigen::Isometry3d pose);
    void logKeypoints(std::vector<cv::Mat> keypoints_3d_vec) const;
    void logLandmarks(std::vector<std::shared_ptr<Frame>> frames) const;
    void logTimecosts(std::vector<int64_t> feature_extraction_costs,
                        std::vector<int64_t> feature_matching_costs,
                        std::vector<int64_t> motion_estimation_costs,
                        std::vector<int64_t> triangulation_costs,
                        std::vector<int64_t> scaling_costs,
                        std::vector<int64_t> optimization_costs,
                        std::vector<int64_t> total_time_costs) const;
    void logTimecostAppend(int64_t feature_extraction_cost,
                        int64_t feature_matching_cost,
                        int64_t motion_estimation_cost,
                        int64_t triangulation_cost,
                        int64_t scaling_cost,
                        int64_t optimization_cost,
                        int64_t total_time_cost);
    void logScales(std::vector<double> scales, std::vector<double> gt_scales) const;
    void logRPE(double rpe_rot, double rpe_trans) const;

    std::string trajectory_file_path_;
    std::string trajectory_file_path_txt_;
    std::string keypoint_file_path_;
    std::string landmark_file_path_;
    std::string cost_file_path_;
    std::string scale_file_path_;
    std::string rpe_file_path_;

    std::ofstream trajectory_file_;
    std::ofstream cost_file_;
};