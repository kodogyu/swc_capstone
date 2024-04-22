#include "tester.hpp"

void Tester::decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat intrinsic, std::vector<cv::Point2f> image0_kp_pts, std::vector<cv::Point2f> image1_kp_pts, const cv::Mat &mask, cv::Mat &_R, cv::Mat &_t) {
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(essential_mat, R1, R2, t);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    std::vector<Eigen::Isometry3d> rel_poses(4, Eigen::Isometry3d::Identity());
    std::vector<int> positive_cnts(4, 0);
    for (int i = 0; i < 4; i++) {
        if (i == 0) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (i == 1) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        else if (i == 2) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (i == 3) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        rel_poses[i].linear() = rotation_mat;
        rel_poses[i].translation() = translation_mat;
        positive_cnts[i] = getPositiveLandmarksCount(intrinsic, image0_kp_pts, image1_kp_pts, rel_poses[i], mask);
        std::cout << "cnt[" << i << "]: " << positive_cnts[i] << std::endl;
        // poses.push_back(rel_poses[i]);
    }

    int max_cnt = 0, max_idx = 0;
    for (int i = 0; i < 4; i++) {
        // std::cout << "cnt[" << i << "]: " << positive_cnts[i] << std::endl;
        if (positive_cnts[i] > max_cnt) {
            max_cnt = positive_cnts[i];
            max_idx = i;
        }
    }
    std::cout << "max idx: " << max_idx << std::endl;

    Eigen::Matrix3d rotation = rel_poses[max_idx].rotation();
    Eigen::Vector3d translation = rel_poses[max_idx].translation();
    cv::eigen2cv(rotation, _R);
    cv::eigen2cv(translation, _t);
}


int Tester::getPositiveLandmarksCount(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask/*, std::vector<Eigen::Vector3d> &landmarks*/) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    int positive_cnt = 0;
    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector3d cam1_point_3d = cam1_pose.inverse().matrix().block<3, 4>(0, 0) * point_3d_homo;
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (cam1_point_3d.z() > 0 && point_3d.z() > 0) {
            positive_cnt++;
        }
    }
    return positive_cnt;
}



