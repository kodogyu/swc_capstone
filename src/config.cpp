#include "config.hpp"

void Configuration::parse() {
    cv::FileStorage config_file(config_path_, cv::FileStorage::READ);
    CV_Assert(config_file.isOpened());

    // Datatset
    num_frames_ = config_file["num_frames"];
    left_images_dir_ = config_file["left_images_dir"];
    getImageEntries();
    gt_path_ = std::string(config_file["gt_path"]);
    is_kitti_ = static_cast<bool>(static_cast<int>(config_file["is_kitti"]));

    // Camera
    fx_ = config_file["fx"];
    fy_ = config_file["fy"];
    s_ = config_file["s"];
    cx_ = config_file["cx"];
    cy_ = config_file["cy"];

    // Visualize
    display_type_ = config_file["display_type"];  // 0: pose only, 1: pose & landmarks
    display_gt_ = static_cast<bool>(static_cast<int>(config_file["display_gt"]));

    // Feature extraction
    num_features_ = config_file["num_features"];

    // Feature matching
    des_dist_thresh_ = config_file["des_dist_thresh"];

    // Optimization
    do_optimize_ = static_cast<bool>(static_cast<int>(config_file["do_optimize"]));
    window_size_ = config_file["window_size"];
    optimizer_verbose_ = static_cast<bool>(static_cast<int>(config_file["optimizer_verbose"]));
}

void Configuration::getImageEntries() {
    std::filesystem::directory_iterator left_images_itr(left_images_dir_);
    std::vector<std::string> left_image_entries_temp;

    // this reads all image entries.
    while (left_images_itr != std::filesystem::end(left_images_itr)) {
        const std::filesystem::directory_entry left_image_entry = *left_images_itr;

        left_image_entries_temp.push_back(left_image_entry.path());

        left_images_itr++;
    }
    // sort entry vectors
    std::sort(left_image_entries_temp.begin(), left_image_entries_temp.end());

    // take only what we want
    left_image_entries_.insert(left_image_entries_.begin(),
                                left_image_entries_temp.begin(),
                                left_image_entries_temp.begin() + num_frames_);
}