#include "timer.hpp"

void Timer::start() {
    start_time_ = std::chrono::steady_clock::now();
}

int64_t Timer::stop() {
    end_time_ = std::chrono::steady_clock::now();
    auto time_diff = end_time_ - start_time_;

    // time cost (ms)
    time_cost_ = std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count();

    return time_cost_;
}