#include "config.hpp"

void Configuration::parse() {
    cv::FileStorage config_file(config_path_, cv::FileStorage::READ);
    CV_Assert(config_file.isOpened());

    // Datatset
    num_frames_ = config_file["num_frames"];
    frame_offset_ = config_file["frame_offset"];
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
    display_type_ = config_file["display_type"];
    display_gt_ = static_cast<bool>(static_cast<int>(config_file["display_gt"]));

    // Feature extraction
    num_features_ = config_file["num_features"];
    filtering_mode_ = config_file["filtering_mode"];
    patch_width_ = config_file["patch_width"];
    patch_height_ = config_file["patch_height"];
    kps_per_patch_ = config_file["kps_per_patch"];

    // Feature matching
    des_dist_thresh_ = config_file["des_dist_thresh"];

    // Optimization
    do_optimize_ = static_cast<bool>(static_cast<int>(config_file["do_optimize"]));
    window_size_ = config_file["window_size"];
    optimizer_verbose_ = static_cast<bool>(static_cast<int>(config_file["optimizer_verbose"]));

    // Test
    calc_reprojection_error_ = static_cast<bool>(static_cast<int>(config_file["calc_reprojection_error"]));
    print_conf_ = static_cast<bool>(static_cast<int>(config_file["print_conf"]));

    if (print_conf_) {
        print();
    }
}

void Configuration::getImageEntries() {
    std::filesystem::directory_iterator left_images_itr(left_images_dir_);
    std::vector<std::string> left_image_entries_temp;
    int frame_cnt = 0;

    // this reads all image entries.
    while (left_images_itr != std::filesystem::end(left_images_itr)) {
        const std::filesystem::directory_entry left_image_entry = *left_images_itr;

        left_image_entries_temp.push_back(left_image_entry.path());

        left_images_itr++;
        frame_cnt++;
    }
    // sort entry vectors
    std::sort(left_image_entries_temp.begin(), left_image_entries_temp.end());

    if (num_frames_ == -1) {
        num_frames_ = frame_cnt - frame_offset_;
    }

    // take only what we want
    left_image_entries_.insert(left_image_entries_.begin(),
                                left_image_entries_temp.begin() + frame_offset_,
                                left_image_entries_temp.begin() + frame_offset_ + num_frames_);
}

void Configuration::print() {
    std::cout << "========== Configuration List ==========" << std::endl;
    // Datatset
    std::cout << "[Dataset]" << std::endl;
    std::cout << "num_frames: " << num_frames_ << "\n"
            << "frame_offset: " << frame_offset_ << "\n"
            << "left_iamges_dir: " << left_images_dir_ << "\n"
            << "gt_path: " << gt_path_ << "\n"
            << "is_kitti: " << is_kitti_ << "\n\n";

    // Camera
    std::cout << "[Camera]" << std::endl;
    std::cout << "fx: " << fx_ << "\n"
            << "fy: " << fy_ << "\n"
            << "s: " << s_ << "\n"
            << "cx: " << cx_ << "\n"
            << "cy: " << cy_ << "\n\n";

    // Visualize
    std::cout << "[Visualize]" << std::endl;
    std::cout << "display_type: " << display_type_ << "\n"
            << "display_gt: " << display_gt_ << "\n\n";


    // Feature extraction
    std::cout << "[Feature Extraction]" << std::endl;
    std::cout << "num_features: " << num_features_ << "\n";
    std::cout << "filtering_mode: " << filtering_mode_ << "\n";
    std::cout << "patch_width: " << patch_width_ << "\n";
    std::cout << "patch_height: " << patch_height_ << "\n";
    std::cout << "kps_per_patch: " << kps_per_patch_ << "\n\n";

    // Feature matching
    std::cout << "[Feature Extraction]" << std::endl;
    std::cout << "des_dist_thresh: " << des_dist_thresh_ << "\n\n";

    // Optimization
    std::cout << "[Optimization]" << std::endl;
    std::cout << "do_optimize: " << do_optimize_ << "\n"
            << "window_size: " << window_size_ << "\n"
            << "optimizer_verbose: " << optimizer_verbose_ << "\n\n";

    // Test
    std::cout << "[Test]" << std::endl;
    std::cout << "calc_reprojection_error: " << calc_reprojection_error_ << "\n\n";
    std::cout << "========================================" << std::endl;
}








