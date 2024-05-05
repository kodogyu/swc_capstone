#include "ransac.hpp"

Ransac::Ransac() {};

Ransac::Ransac(int sample_size, double inlier_prob, double threshold,
                const std::vector<cv::Point2f> &image0_kp_pts,
                const std::vector<cv::Point2f> &image1_kp_pts)
    : sample_size_(sample_size),
      inlier_prob_(inlier_prob),
      threshold_(threshold),
      image0_kp_pts_(image0_kp_pts),
      image1_kp_pts_(image1_kp_pts) {
    sample_pool_size_ = image0_kp_pts_.size();
    // max_iterations_ = std::log(1 - inlier_prob_) / std::log(1 - std::pow(alpha_, sample_size_));
    max_iterations_ = 150;
};

void Ransac::runOnce() {
    int inlier_cnt;

    // sampling
    std::vector<cv::Point2f> image0_kp_pts_samples(sample_size_);
    std::vector<cv::Point2f> image1_kp_pts_samples(sample_size_);
    getSamples(image0_kp_pts_samples, image1_kp_pts_samples);

    // modeling
    Eigen::Matrix3d fundamental_mat;
    getModel(image0_kp_pts_samples, image1_kp_pts_samples, fundamental_mat);

    // calculate inliers
    std::vector<int> inlier_idxes;
    inlier_cnt = getInliers(fundamental_mat, inlier_idxes);

    // get best model
    if (inlier_cnt > max_inlier_cnt_) {
        max_inlier_cnt_ = inlier_cnt;

        best_inlier_idxes_ = inlier_idxes;
        best_model_ = fundamental_mat;
    }
}

void Ransac::run() {
    for (int i = 0; i < max_iterations_; i++) {
        runOnce();
    }
}

void Ransac::getSamples(std::vector<cv::Point2f> &image0_kp_pts_samples, std::vector<cv::Point2f> &image1_kp_pts_samples) {
    // uniform sample indexes
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(0, sample_pool_size_);

    std::vector<int> sample_idxes;
    int sample_cnt = 0;
    while (sample_cnt < sample_size_) {
        int sample_idx = distr(generator);  // random index (uniform)

        if (std::find(sample_idxes.begin(), sample_idxes.end(), sample_idx) == sample_idxes.end()) {
            sample_idxes.push_back(sample_idx);
            sample_cnt++;
        }
    }

    // get samples
    for (int i = 0; i < sample_cnt; i++) {
        int idx = sample_idxes[i];
        image0_kp_pts_samples[i] = image0_kp_pts_[idx];
        image1_kp_pts_samples[i] = image1_kp_pts_[idx];
    }

}

void Ransac::getModel(const std::vector<cv::Point2f> &image0_kp_pts_samples, const std::vector<cv::Point2f> &image1_kp_pts_samples, Eigen::Matrix3d &fundamental_mat) {
    fundamental_mat = Utils().findFundamentalMat(image0_kp_pts_samples, image1_kp_pts_samples);
}

int Ransac::getInliers(const Eigen::Matrix3d &fundamental_mat, std::vector<int> &inlier_idxes) {
    std::vector<int> inlier_idxes_temp;

    // sampson distance
    Eigen::Vector3d l, l_prime;

    for(int i = 0; i < sample_pool_size_; i++) {
        Eigen::Vector3d p(image0_kp_pts_[i].x, image0_kp_pts_[i].y, 1);
        Eigen::Vector3d p_prime(image1_kp_pts_[i].x, image1_kp_pts_[i].y, 1);

        // epiline
        l = fundamental_mat * p_prime;
        l_prime = fundamental_mat.transpose() * p;

        // sampson distance 분자, 분모
        double e = p.transpose() * fundamental_mat * p_prime;
        e = e * e;
        double denominator = l[0] * l[0] + l[1] * l[1] + l_prime[0] * l_prime[0] + l_prime[1] * l_prime[1];

        // sampson distance
        double sampson_distance = e / denominator;

        // inlier
        if (sampson_distance < threshold_) {
            inlier_idxes_temp.push_back(i);
        }
    }
    inlier_idxes = inlier_idxes_temp;

    return inlier_idxes_temp.size();
}
