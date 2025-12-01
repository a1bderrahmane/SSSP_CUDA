#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include "Isolver.cuh" 
#include "CSR.hpp" 


class GPUsolver : public Isolver {
public:
    std::vector<int> solve(const std::string& filename, int source_node) override;
    ~GPUsolver() override = default;
};

#endif // GPU_SOLVER_CUH