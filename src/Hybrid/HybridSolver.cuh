#ifndef HYBRID_SOLVER_CUH
#define HYBRID_SOLVER_CUH

#include "Isolver.cuh" 


class HybridSolver : public Isolver {
public:
    std::vector<int> solve(const std::string& filename, int source_node) override;
    ~HybridSolver() override = default;
};

#endif // HYBRID_SOLVER_CUH