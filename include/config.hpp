#pragma once

#include "common_includes.hpp"

class Configuration {
public:
    Configuration(std::string config_path) {config_path_ = config_path;}
    void parse();
    void getImageEntries();

    // configuration file path
    std::string config_path_;

    // Dataset
    int num_frames_;
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
    int display_type_;
    bool display_gt_;

    // Feature extraction
    int num_features_;

    // Feature matching
    double des_dist_thresh_;

    // Optimization
    bool do_optimize_;
    int window_size_;
    bool optimizer_verbose_;
};