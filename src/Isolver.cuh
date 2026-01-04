#ifndef I_SOLVER_CUH
#define I_SOLVER_CUH

#include <vector>
#include <string>
class Isolver {
public:
    virtual std::vector<int> solve(const std::string& filename, uint source_node) = 0;
    virtual ~Isolver() = default;
};

#endif