#ifndef I_SOLVER_CUH
#define I_SOLVER_CUH

#include <vector>
#include <string>
class Isolver {
public:
    virtual std::vector<uint> solve(uint source_node) = 0;
    virtual ~Isolver() = default;
};

#endif