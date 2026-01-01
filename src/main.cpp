#include "GPU/GPUsolver.cuh"
#include <string>
#include <vector>
int main() {
    string filename="datasets/simple-graph.txt";
    GPUsolver solver = GPUsolver(filename);
    solver.solve(filename,1);
    std::vector<uint> result=solver.getDistancesHost();
    return 0;
}
