#include <sstream>
#include <fstream>

#include "common_includes.hpp"
#include "frame.hpp"

class Utils {
public:
    Utils();

    void drawFramesLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);
    void drawReprojectedLandmarks(const std::vector<std::shared_ptr<Frame>> &frames);

    std::vector<Eigen::Isometry3d> calcRPE(const std::string gt_path, const std::vector<std::shared_ptr<Frame>> &frames);
    void calcRPE_rt(const std::string gt_path, const std::vector<std::shared_ptr<Frame>> &frames, double &_rpe_rot, double &_rpe_trans);
    void loadGT(std::string gt_path, std::vector<Eigen::Isometry3d> &_gt_poses);

    double calcReprojectionError(const std::vector<std::shared_ptr<Frame>> &frames);

    void drawCorrespondingFeatures(const std::vector<std::shared_ptr<Frame>> &frames, const int pivot_frame_idx, const int dup_count);
};