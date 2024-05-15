#include "visual_odometry.hpp"
#include "tester.hpp"

int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;
    if (argc != 2) {
        std::cout << "Usage: visual_odometry config_yaml" << std::endl;
        return 1;
    }

    VisualOdometry vo(argv[1]);

    if (vo.pConfig_->test_mode_) {  // test mode
        Tester tester;
        tester.run(vo);
    }
    else {  // normal mode
        vo.run();
    }

    return 0;
}