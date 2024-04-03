#include "common_includes.hpp"
#include "visualizer.hpp"

int main() {
    Visualizer vis;
    vis.displayPoses(std::vector<gtsam::Pose3>());

    return 0;
}