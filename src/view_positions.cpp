#include <fstream>
#include <sstream>

#include "visualizer.hpp"

int main() {
    std::ifstream traj_file("traj.txt");
    std::string line;
    int x, y;
    std::vector<std::pair<int, int>> positions;
    while(std::getline(traj_file, line)) {
        std::stringstream ssline(line);
        ssline >> x >> y;
        positions.push_back(std::pair(x, y));
    }

    Visualizer vis;
    vis.drawPositions(positions);

}