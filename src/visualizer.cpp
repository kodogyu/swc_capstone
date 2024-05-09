#include "visualizer.hpp"

Visualizer::Visualizer(std::shared_ptr<Configuration> pConfig, std::shared_ptr<Utils> pUtils) {
    pConfig_ = pConfig;
    pUtils_ = pUtils;

    newest_pointer_ = 0;

    if (pConfig_->display_type_ == DisplayType::REALTIME_VIS) {
        visualizer_thread_ = std::thread(std::bind(&Visualizer::displayPoseRealtime, this));
    }
}

Visualizer::~Visualizer() {
    if (pConfig_->display_type_ == DisplayType::REALTIME_VIS) {
        visualizer_thread_.join();
    }
}


void Visualizer::displayPoses(const std::vector<Eigen::Isometry3d> &poses) {
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

    std::vector<Eigen::Isometry3d> gt_poses;
    pUtils_->loadGT(gt_poses);

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

        if (pConfig_->display_gt_) {
            drawGT(gt_poses);
        }
        pangolin::FinishFrame();
    }
}

void Visualizer::displayPoses(const std::vector<std::shared_ptr<Frame>> &frames) {
    std::vector<Eigen::Isometry3d> poses;
    for (auto pFrame: frames) {
        poses.push_back(pFrame->pose_);
    }

    displayPoses(poses);
}

void Visualizer::displayPoseWithKeypoints(const std::vector<Eigen::Isometry3d> &poses, const std::vector<cv::Mat> &keypoints_3d_vec) {
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

    const float red[3] = {1, 0, 0};
    const float orange[3] = {1, 0.5, 0};
    const float yellow[3] = {1, 1, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    const float navy[3] = {0, 0.02, 1};
    const float purple[3] = {0.5, 0, 1};
    std::vector<const float*> colors {red, orange, yellow, green, blue, navy, purple};


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

        int color_idx = 0;
        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);
        for (int i = 0; i < poses.size(); i++) {
            auto cam_pose = poses[i];
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
            glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            // draw map points in world coordinate
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (int j = 0; j < keypoints_3d_vec[i].cols; j++) {
                double point[3] = {keypoints_3d_vec[i].at<double>(0, j),
                                    keypoints_3d_vec[i].at<double>(1, j),
                                    keypoints_3d_vec[i].at<double>(2, j)};

                glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
                glVertex3d(point[0], point[1], point[2]);
            }
            glEnd();

            last_center = Ow;

            color_idx++;
            color_idx = color_idx % colors.size();
        }

        pangolin::FinishFrame();
    }
}

void Visualizer::displayFramesAndLandmarks(const std::vector<std::shared_ptr<Frame>> &frames) {
    pangolin::CreateWindowAndBind("Visual Odometry Viewer", 1024, 768);
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

    const float red[3] = {1, 0, 0};
    const float orange[3] = {1, 0.5, 0};
    const float yellow[3] = {1, 1, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    const float navy[3] = {0, 0.02, 1};
    const float purple[3] = {0.5, 0, 1};
    std::vector<const float*> colors {red, orange, yellow, green, blue, navy, purple};

    std::vector<Eigen::Isometry3d> gt_poses;
    pUtils_->loadGT(gt_poses);

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

        int color_idx = 0;
        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);
        for (auto pFrame : frames) {
            auto cam_pose = pFrame->pose_;
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
            glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            // draw map points in world coordinate
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (auto pLandmark : pFrame->landmarks_) {
                Eigen::Vector3d point = pLandmark->point_3d_;

                glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
                glVertex3d(point[0], point[1], point[2]);
            }
            glEnd();

            last_center = Ow;

            color_idx++;
            color_idx = color_idx % colors.size();
        }

        if (pConfig_->display_gt_) {
            drawGT(gt_poses);
        }
        pangolin::FinishFrame();
    }
}


// display online
void Visualizer::displayPoseRealtime() {
    pangolin::CreateWindowAndBind("Visual Odometry Viewer", 1024, 768);
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

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        // lock buffer mutex
        std::lock_guard<std::mutex> lock(buffer_mutex_);

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
        for (const Eigen::Isometry3d &cam_pose : est_pose_buffer_) {
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

        if (pConfig_->display_gt_ && newest_pointer_ > 0) {
            drawGT(gt_buffer_);
        }

        if (newest_pointer_ > 0) {
            cv::imshow("Frame viewer", current_frame_->image_);
            cv::waitKey(1);
        }

        pangolin::FinishFrame();
    }
}

//! TODO: displayPoseAndLandmarksRealtime()

void Visualizer::updateBuffer(const std::shared_ptr<Frame> &pFrame) {
    // std::cout << "----- Visualizer::updateBuffer -----" << std::endl;
    // lock buffer mutex
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // first frame
    if (newest_pointer_ == 0) {
        std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();
        est_pose_buffer_.push_back(pPrev_frame->pose_);
        gt_buffer_.push_back(pUtils_->getGT(pPrev_frame->frame_image_idx_));
    }
    est_pose_buffer_.push_back(pFrame->pose_);
    gt_buffer_.push_back(pUtils_->getGT(pFrame->frame_image_idx_));

    current_frame_ = pFrame;

    newest_pointer_++;
}

void Visualizer::updateBuffer(const std::vector<std::shared_ptr<Frame>> &frames) {

    // lock buffer mutex
    std::lock_guard<std::mutex> lock(buffer_mutex_);


    // first frame
    if (newest_pointer_ == 0) {
        std::shared_ptr<Frame> pFirst_frame = frames[0]->pPrevious_frame_.lock();
        est_pose_buffer_.push_back(pFirst_frame->pose_);
        gt_buffer_.push_back(pUtils_->getGT(pFirst_frame->frame_image_idx_));
    }

    // fix optimized poses
    if (frames.size() == pConfig_->window_size_) {
        for (int i = 0; i < frames.size() - 1; i++) {
            int buffer_idx = frames[i]->id_;
            est_pose_buffer_[buffer_idx] = frames[i]->pose_;
        }
    }
    est_pose_buffer_.push_back(frames[frames.size() - 1]->pose_);
    gt_buffer_.push_back(pUtils_->getGT(frames[frames.size() - 1]->frame_image_idx_));

    current_frame_ = frames[frames.size() - 1];

    newest_pointer_++;
}


// display offline
void Visualizer::display(int display_type) {
    pangolin::CreateWindowAndBind("Visual Odometry Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ----- Display Panel -----
    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -3, -3, 0, 0, 0, 0.0, -1.0, 0.0));

    // Choose a sensible left UI Panel width based on the width of 20
    // charectors from the default font.
    const int UI_WIDTH = 20* pangolin::default_font().MaxWidth();

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    // ----- UI Panel-----
    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    // checkboxes
    pangolin::Var<bool> checkbox_poses("ui.Draw Poses",false,true);
    pangolin::Var<bool> checkbox_landmarks("ui.Draw Landmarks",false,true);
    pangolin::Var<bool> checkbox_keypoints("ui.Draw Keypoints",false,true);
    pangolin::Var<bool> checkbox_gt("ui.Draw GT",false,true);

    // std::function objects can be used for Var's too. These work great with C++11 closures.
    pangolin::Var<std::function<void(void)>> save_window("ui.Save Window", [](){
        pangolin::SaveWindowOnRender("output_logs/view_images/window");
    });

    pangolin::Var<std::function<void(void)>> save_cube_view("ui.Save Trajectory view", [&vis_display](){
        pangolin::SaveWindowOnRender("output_logs/view_images/trajectory", vis_display.v);
    });

    // initialize checkboxes
    if (display_type & DisplayType::POSE) {
        checkbox_poses = true;
    }
    if (display_type & DisplayType::LANDMARKS) {
        checkbox_landmarks = true;
    }
    if (display_type & DisplayType::KEYPOINTS) {
        checkbox_keypoints = true;
    }
    if (pConfig_->display_gt_) {
        checkbox_gt = true;
    }

    // ----- Draw Trajectory -----
    // colors
    const float red[3] = {1, 0, 0};
    const float orange[3] = {1, 0.5, 0};
    const float yellow[3] = {1, 1, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    const float navy[3] = {0, 0.02, 1};
    const float purple[3] = {0.5, 0, 1};
    const float black[3] = {0, 0, 0};
    std::vector<const float*> colors {black, red, orange, yellow, green, blue, navy, purple};
    // std::vector<const float*> colors {black, red, green, orange, yellow, blue, navy, purple};

    // load gt trajectory
    std::vector<Eigen::Isometry3d> gt_poses;
    pUtils_->loadGT(gt_poses);

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

        // Draw data
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);
        drawPose(frame_buffer_[0], colors[0], last_center);  // first frame (원점)

        int color_idx = 1;
        for (int i = 1; i < frame_buffer_.size(); i++) {
            std::shared_ptr<Frame> pFrame = frame_buffer_[i];
            // draw camera pose
            if (checkbox_poses) {
                drawPose(pFrame, colors[color_idx], last_center);
            }

            // draw map points in world coordinate
            if (checkbox_landmarks) {
                drawLandmarks(pFrame, colors[color_idx]);
            }

            // draw keypoints_3d in world coordinate
            if (checkbox_keypoints) {
                drawKeypoints(pFrame, colors[color_idx]);
            }

            //! TODO covisible landmarks only

            last_center = pFrame->pose_.translation();
            color_idx++;
            color_idx = color_idx % colors.size();
        }

        // GT trajectory
        if (checkbox_gt) {
            drawGT(gt_poses);
        }

        pangolin::FinishFrame();
    }
}

void Visualizer::drawPose(const std::shared_ptr<Frame> &pFrame, const float color[], const Eigen::Vector3d& last_center) {
    auto cam_pose = pFrame->pose_;
    Eigen::Vector3d Ow = cam_pose.translation();
    Eigen::Vector3d Xw = cam_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::Vector3d Yw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
    Eigen::Vector3d Zw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));

    glBegin(GL_LINES);
    // X
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Xw[0], Xw[1], Xw[2]);

    // Y
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Yw[0], Yw[1], Yw[2]);

    // Z
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Zw[0], Zw[1], Zw[2]);
    glEnd();

    // draw odometry line
    glBegin(GL_LINES);
    glColor3f(color[0], color[1], color[2]);
    glVertex3d(last_center[0], last_center[1], last_center[2]);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glEnd();
}

void Visualizer::drawLandmarks(const std::shared_ptr<Frame> &pFrame, const float color[]) {
    glPointSize(5.0f);
    glBegin(GL_POINTS);

    Eigen::Vector3d point;
    for (auto pLandmark : pFrame->landmarks_) {
        point = pLandmark->point_3d_;

        glColor3f(color[0], color[1], color[2]);
        glVertex3d(point[0], point[1], point[2]);
    }
    glEnd();
}

void Visualizer::drawKeypoints(const std::shared_ptr<Frame> &pFrame, const float color[]) {
    glPointSize(5.0f);
    glBegin(GL_POINTS);

    for (auto keypoint_3d : pFrame->keypoints_3d_) {
        glColor3f(color[0], color[1], color[2]);
        glVertex3d(keypoint_3d[0], keypoint_3d[1], keypoint_3d[2]);
    }

    glEnd();
}

void Visualizer::drawGT(const std::vector<Eigen::Isometry3d> &_gt_poses) {
    Eigen::Isometry3d first_pose = _gt_poses[0];
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

void Visualizer::drawPositions(const std::vector<std::pair<int, int>> &positions) {
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

        std::pair<int, int> first_pose = positions[0];
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);

        for(auto position : positions) {
            Eigen::Vector3d Ow = Eigen::Vector3d(position.first, position.second, 0);
            Eigen::Vector3d Xw = (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
            Eigen::Vector3d Yw = (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
            Eigen::Vector3d Zw = (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
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
}

void drawKeypoints3DCorrespondingLandmark(const std::shared_ptr<Frame> &pFrame, const float color[]) {
    glPointSize(5.0f);
    glBegin(GL_POINTS);

    int frame_id = pFrame->id_;
    int keypoint_idx;
    Eigen::Vector3d keypoint_3d;
    for (int i = 0; i < pFrame->landmarks_.size(); i++) {
        keypoint_3d = pFrame->keypoints_3d_[i];

        glColor3f(color[0], color[1], color[2]);
        glVertex3d(keypoint_3d[0], keypoint_3d[1], keypoint_3d[2]);
    }


    glEnd();
}


void Visualizer::setFrameBuffer(const std::vector<std::shared_ptr<Frame>> &frames) {
    frame_buffer_ = frames;
}