#ifndef I_SOLVER_CUH
#define I_SOLVER_CUH

#include <vector> 
#include "GPU/graph.hpp"
#include <cuda_runtime.h>


class Isolver {
public:
    virtual std::vector<int> solve(const Graph& graph, int source_node) = 0;
    virtual ~Isolver() = default;
};

#endif // I_SOLVER_CUH