#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <pangolin/pangolin.h>

void displayPoses(const std::vector<gtsam::Pose3> &poses);
void loadGT(std::vector<gtsam::Pose3> &_gt_poses);
void drawGT(const std::vector<gtsam::Pose3> &_gt_poses);

int main(int argc, char** argv) {
    //**========== 0. Image load ==========**//
    if (argc != 2) {
        std::cout << "Usage: visual_odometry_example config_yaml" << std::endl;
        return 1;
    }
    double fx = 788.629315;
    double fy = 786.382230;
    double s = 0.0;
    double cx = 687.158398;
    double cy = 317.752196;
    double baseline = 0.008;  // baseline = 0.008m (=8mm)
    cv::Mat cameraMatrix(cv::Size(3, 3), CV_64F);
    cameraMatrix.at<double>(0, 0) = fx;
    cameraMatrix.at<double>(0, 1) = 0;
    cameraMatrix.at<double>(0, 2) = cx;
    cameraMatrix.at<double>(1, 0) = 0;
    cameraMatrix.at<double>(1, 1) = fy;
    cameraMatrix.at<double>(1, 2) = cy;
    cameraMatrix.at<double>(2, 0) = 0;
    cameraMatrix.at<double>(2, 1) = 0;
    cameraMatrix.at<double>(2, 2) = 1;

    cv::FileStorage config_file(argv[1], cv::FileStorage::READ);
    int num_frames = config_file["num_frames"];
    std::vector<std::string> left_image_entries;
    std::filesystem::path left_images_dir(config_file["left_images_dir"]);
    std::filesystem::directory_iterator left_images_itr(left_images_dir);

    // this reads all image entries. Therefore length of the image entry vector may larger than the 'num_frames'
    while (left_images_itr != std::filesystem::end(left_images_itr)) {
        const std::filesystem::directory_entry left_image_entry = *left_images_itr;

        left_image_entries.push_back(left_image_entry.path());

        left_images_itr++;
    }
    // sort entry vectors
    std::sort(left_image_entries.begin(), left_image_entries.end());

    cv::Mat image0_left, image1_left;
    gtsam::Pose3 relative_pose;
    std::vector<gtsam::Pose3> poses;
    // read images
    image0_left = cv::imread(left_image_entries[0], cv::IMREAD_GRAYSCALE);
    poses.push_back(gtsam::Pose3());
    for (int i = 1; i < num_frames; i++) {
        image1_left = cv::imread(left_image_entries[i], cv::IMREAD_GRAYSCALE);

        //**========== 1. Feature extraction ==========**//
        cv::Mat image0_left_descriptors, image1_left_descriptors;
        std::vector<cv::KeyPoint> image0_left_keypoints, image1_left_keypoints;
        // create orb feature extractor
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        orb->detectAndCompute(image0_left, cv::Mat(), image0_left_keypoints, image0_left_descriptors);
        orb->detectAndCompute(image1_left, cv::Mat(), image1_left_keypoints, image1_left_descriptors);

        //TODO matched keypoint filtering (RANSAC?)
        //**========== 2. Feature matching ==========**//
        // create a matcher
        cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

        // image0 left & image1 left (matcher matching)
        std::vector<std::vector<cv::DMatch>> image_matches_vec;
        double between_dist_thresh = 0.60;
        orb_matcher->knnMatch(image0_left_descriptors, image1_left_descriptors, image_matches_vec, 2);

        std::vector<cv::DMatch> good_matches;  // good matchings
        for (int i = 0; i < image_matches_vec.size(); i++) {
            if (image_matches_vec[i][0].distance < image_matches_vec[i][1].distance * between_dist_thresh) {
                good_matches.push_back(image_matches_vec[i][0]);
            }
        }
        std::cout << "original features for image1&2: " << image_matches_vec.size() << std::endl;
        std::cout << "good features for image1&2: " << good_matches.size() << std::endl;

        cv::Mat image_matches;
        cv::drawMatches(image0_left, image0_left_keypoints, image1_left, image1_left_keypoints, good_matches, image_matches);
        cv::imwrite("output_logs/inter_frames/frame"
                + std::to_string(i - 1)
                + "&"
                + std::to_string(i)
                + "_kp_matches(raw).png", image_matches);

        // RANSAC
        std::vector<cv::Point> image0_kp_pts;
        std::vector<cv::Point> image1_kp_pts;
        for (auto match : good_matches) {
            image0_kp_pts.push_back(image0_left_keypoints[match.queryIdx].pt);
            image1_kp_pts.push_back(image1_left_keypoints[match.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat fundamental_mat = cv::findFundamentalMat(image0_kp_pts, image1_kp_pts, mask, cv::FM_RANSAC, 3.0, 0.99);
        cv::Mat ransac_matches;
        cv::drawMatches(image0_left, image0_left_keypoints,
                        image1_left, image1_left_keypoints,
                        good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask);
        cv::imwrite("output_logs/inter_frames/frame"
                + std::to_string(i - 1)
                + "&"
                + std::to_string(i)
                + "_kp_matches(ransac).png", ransac_matches);

        //** Motion estimation **//
        cv::Mat essential_mat = cameraMatrix.t() * fundamental_mat * cameraMatrix;;
        cv::Mat R, t;
        cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, cameraMatrix, R, t, mask);

        Eigen::Matrix3d rotation_mat;
        Eigen::Vector3d translation_mat;
        cv::cv2eigen(R, rotation_mat);
        cv::cv2eigen(t, translation_mat);
        relative_pose = gtsam::Pose3(gtsam::Rot3(rotation_mat), gtsam::Point3(translation_mat));
        poses.push_back(relative_pose * poses[i - 1]);

        // move on
        image0_left = image1_left;
    }


    //** Log **//
    std::ofstream log_file("output_logs/trajectory.csv");
    log_file << "qw,qx,qy,qz,x,y,z\n";
    for (auto pose : poses) {
        gtsam::Vector quaternion = pose.rotation().quaternion();
        gtsam::Vector position = pose.translation();
        log_file << quaternion.w() << "," << quaternion.x() << "," << quaternion.y() << "," << quaternion.z() << ","
                    << position.x() << "," << position.y() << "," << position.z() << "\n";
    }

    //** Visualize **//
    displayPoses(poses);

    return 0;
}

void displayPoses(const std::vector<gtsam::Pose3> &poses) {
    pangolin::CreateWindowAndBind("Visual Odometry Example", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -3, -3, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    std::vector<gtsam::Pose3> gt_poses;
    loadGT(gt_poses);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        // draw the original axis
        glLineWidth(3);
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);

        for (auto cam_pose : poses) {
            Eigen::Vector3d Ow = cam_pose.translation();
            Eigen::Vector3d Xw = cam_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
            Eigen::Vector3d Yw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
            Eigen::Vector3d Zw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
            // draw odometry line
            glBegin(GL_LINES);
            glColor3f(0.0, 0.0, 0.0);
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            last_center = Ow;
        }

        drawGT(gt_poses);
        pangolin::FinishFrame();
    }
}

void loadGT(std::vector<gtsam::Pose3> &_gt_poses) {
    std::ifstream gt_poses_file("/home/kodogyu/Datasets/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt");
    int no_frame;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;
    std::vector<gtsam::Pose3> gt_poses;

    while(std::getline(gt_poses_file, line)) {
        std::stringstream ssline(line);
        ssline >> no_frame
                >> r11 >> r12 >> r13 >> t1
                >> r21 >> r22 >> r23 >> t2
                >> r31 >> r32 >> r33 >> t3;

        Eigen::Matrix3d rotation_mat;
        rotation_mat << r11, r12, r13,
                        r21, r22, r23,
                        r31, r32, r33;
        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;
        gtsam::Pose3 gt_pose = gtsam::Pose3(gtsam::Rot3(rotation_mat), gtsam::Point3(translation_mat));
        gt_poses.push_back(gt_pose);
    }

    _gt_poses = gt_poses;
}

void drawGT(const std::vector<gtsam::Pose3> &_gt_poses) {
    gtsam::Pose3 first_pose = _gt_poses[0];
    Eigen::Vector3d last_center(0.0, 0.0, 0.0);

    for(auto gt_pose : _gt_poses) {
        gt_pose = first_pose.inverse() * gt_pose;
        Eigen::Vector3d Ow = gt_pose.translation();
        Eigen::Vector3d Xw = gt_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
        Eigen::Vector3d Yw = gt_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
        Eigen::Vector3d Zw = gt_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Xw[0], Xw[1], Xw[2]);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Yw[0], Yw[1], Yw[2]);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Zw[0], Zw[1], Zw[2]);
        glEnd();
        // draw odometry line
        glBegin(GL_LINES);
        glColor3f(0.0, 0.0, 1.0); // blue
        glVertex3d(last_center[0], last_center[1], last_center[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glEnd();

        last_center = Ow;
    }

}