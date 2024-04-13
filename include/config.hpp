#pragma once

#include "common_includes.hpp"

enum DisplayType {
    POSE_ONLY,
    POSE_AND_LANDMARKS,
    ALIGNED_POSE,
    REALTIME_VIS
};

enum FilterMode {
    KEYPOINT_FILTERING,
    MATCH_FILTERING
};

class Configuration {
public:
    Configuration(std::string config_path) {config_path_ = config_path;}

    void parse();
    void getImageEntries();
    void print();

    // configuration file path
    std::string config_path_;

    // Dataset
    int num_frames_;
    int frame_offset_;
    std::filesystem::path left_images_dir_;
    std::vector<std::string> left_image_entries_;
    std::vector<std::string> right_image_entries_;
    std::string gt_path_;
    bool is_kitti_;

    // Camera
    double fx_;
    double fy_;
    double s_;
    double cx_;
    double cy_;

    // Visualize
    int display_type_;  // 0: pose only, 1: pose & landmarks, 2: pose only (aligned with gt)
    bool display_gt_;

    // Feature extraction
    int num_features_;
    int filtering_mode_;
    int patch_width_;
    int patch_height_;
    int kps_per_patch_;

    // Feature matching
    double des_dist_thresh_;

    // Optimization
    bool do_optimize_;
    int window_size_;
    bool optimizer_verbose_;

    // Test
    bool calc_reprojection_error_;
    bool print_conf_;
};