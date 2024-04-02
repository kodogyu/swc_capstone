#pragma once

#include "common_includes.hpp"

class Landmark {
public:
    Landmark() {id++;}

    static int id;
    std::map<int, int> observations;  // observations.search(frame_id) = keypoint_index
    gtsam::Point3 point_3d;  // 3D coordinate at corresponding pose
};