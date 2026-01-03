#include <iostream>
#include <string>
#include <vector>
#include "CPUSolver.hpp"

int main() {
    std::string filename="datasets/test-graph.txt";
    CPUSolver solver(filename);
    solver.solve(0);
    solver.printResults();
    return 0;
}