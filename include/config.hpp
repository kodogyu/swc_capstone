#pragma once

#include "common_includes.hpp"

enum DisplayType {
    POSE = 1,
    LANDMARKS = 2,
    KEYPOINTS = 4,
    ALIGNED_POSE = 8,
    REALTIME_VIS = 16,

    POSE_AND_LANDMARKS = POSE + LANDMARKS,  // POSE + LANDMARKS
    POSE_AND_KEYPOINTS = POSE + KEYPOINTS,  // POSE + KEYPOINTS
};

enum FilterMode {
    NO_FILTERING = 0,
    KEYPOINT_FILTERING = 1,
    MATCH_FILTERING = 2
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
    bool is_fisheye_;
    double fx_;
    double fy_;
    double s_;
    double cx_;
    double cy_;
    double xi_ = 0;
    double k1_ = 0, k2_ = 0, p1_ = 0, p2_ = 0, k3_ = 0;

    // Visualize
    int display_type_;  // 0: pose only, 1: pose & landmarks, 2: pose only (aligned with gt)
    bool display_gt_;

    // Feature extraction
    int feature_extractor_;
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
    bool test_mode_;
    bool calc_reprojection_error_;
    bool print_conf_;
};