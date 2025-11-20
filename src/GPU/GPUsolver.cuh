#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include "Isolver.cuh" 
#include "Graph.hpp" 


class GPUsolver : public Isolver {
public:
    std::vector<int> solve(const Graph& graph, int source_node) override;
    ~CPUsolver() override = default;
};

#endif // GPU_SOLVER_CUH