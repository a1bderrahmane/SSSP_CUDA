#ifndef HYBRID_SOLVER_CUH
#define HYBRID_SOLVER_CUH

#include "Isolver.cuh" 
#include "Graph.hpp" 


class HybridSolver : public Isolver {
public:
    std::vector<int> solve(const Graph& graph, int source_node) override;
    ~HybridSolver() override = default;
};

#endif // HYBRID_SOLVER_CUH