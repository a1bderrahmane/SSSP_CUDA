#ifndef CPU_SOLVER_CUH
#define CPU_SOLVER_CUH

#include <atomic>
#include <mutex>

#include "Isolver.hpp"
#include "CSRGraph.hpp"
#include "ConcurrentQueue.cpp"

class CPUSolver : public Isolver
{
public:
    std::vector<int> solve(const std::string &filename, uint source_node) override;
    CPUSolver(const std::string &filename);
    ~CPUSolver() override;

private:
    CSRGraph *csr_graph;
    uint* row_ptr;
    uint* col_idx;
    uint* weights;

    uint *distances;
    uint *predecessors;
    std::mutex outputWriterMutex;

    ConcurrentQueue<uint> *currentQueue;
    ConcurrentQueue<uint> *nextQueue;
    bool* verticesUpdated;

    void solveIteration(uint source_node);
    void solveIterationThreadWork(uint source_node);
    void updateOutputThreadWork(uint vertex, uint neighboor, uint newDistance);
};

#endif // CPU_SOLVER_CPP