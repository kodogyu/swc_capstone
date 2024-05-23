#include <iostream>

#include "visualizer.hpp"

void testModule() {
    // gaussian noise
    const double mean = 0.0;
    const double stddev = 0.0;
    std::default_random_engine generator;
    std::normal_distribution<float> dist(mean, stddev);
    std::normal_distribution<float> dist_3d(mean, 0.4);
    std::cout << "[gaussian noise] mean: " << mean << ", stddev: " << stddev << std::endl;

    // object points
    std::vector<Eigen::Vector3d> object_points, noised_object_points;
    // --- random pattern
    // object_points = {Eigen::Vector3d(15, -20, 35), Eigen::Vector3d(15, -10, 35), Eigen::Vector3d(10, -20, 40), Eigen::Vector3d(10, -10, 40),
    //                                                 Eigen::Vector3d(-10, -20, 40), Eigen::Vector3d(-10, -10, 40), Eigen::Vector3d(-15, -20, 35), Eigen::Vector3d(-15, -10, 35)};
    // --- checkerboard
    int board_row = 8, board_col = 6, stride = 1;
    for (int r = 0; r < board_row; r++) {
        for (int c = 0; c < board_col; c++) {
            Eigen::Vector3d object_point(-(board_col-1)*stride/2 + stride*c, -(board_row-1)*stride/2 + stride*r, 30);
            object_points.push_back(object_point);
            noised_object_points.push_back(object_point + Eigen::Vector3d(0, 0, dist_3d(generator)));
        }
    }
    int board_row2 = 5, board_col2 = 7, stride2 = 1;
    for (int r = 0; r < board_row2; r++) {
        for (int c = 0; c < board_col2; c++) {
            Eigen::Vector3d object_point(-(board_col2-1)*stride2/2 + stride2*c + 10, -(board_row2-1)*stride2/2 + stride2*r, 40);
            object_points.push_back(object_point);
            noised_object_points.push_back(object_point + Eigen::Vector3d(0, 0, dist_3d(generator)));
        }
    }

    std::vector<cv::Point3d> object_points_cv, noised_object_points_cv;
    for (int i = 0; i < object_points.size(); i++) {
        Eigen::Vector3d point = object_points[i];
        object_points_cv.push_back(cv::Point3d(point.x(), point.y(), point.z()));

        Eigen::Vector3d noised_point = noised_object_points[i];
        noised_object_points_cv.push_back(cv::Point3d(noised_point.x(), noised_point.y(), noised_point.z()));
    }

    // intrinsic matrix
    float intrinsic_elems[] = { 350., 0.,   320.,
                                0.,   350., 240.,
                                0.,   0.,   1.};
    cv::Mat intrinsic_cv(3, 3, CV_32FC1, intrinsic_elems);

    Eigen::Matrix3d intrinsic;
    cv::cv2eigen(intrinsic_cv, intrinsic);

    // camera poses
    Eigen::Isometry3d cam0_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d cam1_pose = Eigen::Isometry3d::Identity();
    cam1_pose.translation() = Eigen::Vector3d(0, 0, 10);

    // camera projection matrices
    Eigen::MatrixXd cam0_projection_mat(3, 4);
    cam0_projection_mat = intrinsic * cam0_pose.inverse().matrix().block(0, 0, 3, 4);
    Eigen::MatrixXd cam1_projection_mat(3, 4);
    cam1_projection_mat = intrinsic * cam1_pose.inverse().matrix().block(0, 0, 3, 4);

    // project points
    std::vector<cv::Point2f> cam0_image_points, cam1_image_points;
    for (int i = 0; i < object_points.size(); i++) {
        Eigen::Vector3d object_point = object_points[i];
        Eigen::Vector4d object_point_homo(object_point.x(), object_point.y(), object_point.z(), 1);

        Eigen::Vector3d cam0_image_point_homo = cam0_projection_mat * object_point_homo;
        Eigen::Vector3d cam1_image_point_homo = cam1_projection_mat * object_point_homo;

        cv::Point2f cam0_image_point = cv::Point2f(cam0_image_point_homo.x() / cam0_image_point_homo.z(), cam0_image_point_homo.y() / cam0_image_point_homo.z());
        cv::Point2f cam1_image_point = cv::Point2f(cam1_image_point_homo.x() / cam1_image_point_homo.z(), cam1_image_point_homo.y() / cam1_image_point_homo.z());

        // add gaussian noise
        cam0_image_point += cv::Point2f(dist(generator), dist(generator));
        cam1_image_point += cv::Point2f(dist(generator), dist(generator));

        std::cout << "cam0_image_point: " << cam0_image_point << ", cam1_image_point: " << cam1_image_point << std::endl;

        cam0_image_points.push_back(cam0_image_point);
        cam1_image_points.push_back(cam1_image_point);
    }

    // make image0 & image1
    cv::Mat image0(480, 640, CV_8UC3, cv::Scalar(255, 255, 255)), image1(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < cam0_image_points.size(); i++) {
        cv::Point2f cam0_image_point = cam0_image_points[i];
        cv::Point2f cam1_image_point = cam1_image_points[i];

        cv::circle(image0, cam0_image_point, 3, cv::Scalar(0, 0, 255), 1);
        cv::circle(image1, cam1_image_point, 3, cv::Scalar(0, 0, 255), 1);
    }

    // solve PnP
    cv::Mat rvec, tvec;
    // cv::solvePnP(object_points_cv, cam1_image_points, intrinsic_cv, cv::Mat(), rvec, tvec);
    cv::solvePnP(noised_object_points_cv, cam1_image_points, intrinsic_cv, cv::Mat(), rvec, tvec);

    cv::Mat R, t;
    cv::Rodrigues(rvec, R);
    t = tvec;

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    cv::cv2eigen(R, rotation_mat);
    cv::cv2eigen(t, translation_mat);

    Eigen::Isometry3d relative_pose;
    relative_pose.linear() = rotation_mat;
    relative_pose.translation() = translation_mat;
    relative_pose = relative_pose.inverse();

    std::cout << "relative_pose:\n" << relative_pose.matrix() << std::endl;
    std::cout << "GT cam1_pose:\n" << cam1_pose.matrix() << std::endl;

    // Visualize
    cv::imshow("image0", image0);
    cv::imshow("image1", image1);
    cv::waitKey(0);

    std::vector<Eigen::Isometry3d> poses = {cam0_pose, relative_pose};
    Visualizer visualizer;
    // visualizer.displayPosesAnd3DPoints(poses, object_points);
    visualizer.displayPosesAnd3DPoints(poses, noised_object_points);

    cv::destroyAllWindows();
}

int main() {
    std::cout << "test module" << std::endl;

    testModule();
}