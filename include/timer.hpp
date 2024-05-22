#pragma once

#include <chrono>

class Timer {
public:
    Timer(){};

    void start();
    int64_t stop();

    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::time_point<std::chrono::steady_clock> end_time_;
    int64_t time_cost_;
};