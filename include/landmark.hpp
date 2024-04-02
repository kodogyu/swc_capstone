#pragma once

#include "common_includes.hpp"

class Landmark {
public:
    Landmark() {id_ = total_landmark_cnt_++;}

    static int total_landmark_cnt_;

    int id_;
    std::map<int, int> observations_;  // observations.find(curr_frame.id_) = std::map::itr(curr_frame.id_, keypoint_idx)
    gtsam::Point3 point_3d_;  // 3D coordinate at corresponding pose
};