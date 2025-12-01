#ifndef CPU_SOLVER_CUH
#define CPU_SOLVER_CUH

#include "Isolver.cuh" 


class CPUsolver : public Isolver {
public:
    std::vector<int> solve(const std::string& filename, int source_node) override;
    ~CPUsolver() override = default;
};

#endif // CPU_SOLVER_CUH