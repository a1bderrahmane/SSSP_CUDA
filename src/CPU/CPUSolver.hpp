#ifndef CPU_SOLVER_CUH
#define CPU_SOLVER_CUH

#include <atomic>
#include <mutex>
#include <thread>

#include "Isolver.cuh"
#include "CSR.hpp"
#include "ConcurrentQueue.cpp"

#define NB_THREADS 8


class CPUSolver
{
public:
    void solve(uint source_node);
    void printResults();
    CPUSolver(const std::string &filename);
    ~CPUSolver();

private:
    CSR *csr_graph;
    uint* row_ptr;
    uint* col_idx;
    uint8_t* weights;

    uint *distances;
    uint *predecessors;
    std::mutex outputWriterMutex;

    ConcurrentQueue *vertexQueue;
    bool* verticesUpdated;
    std::thread* threadPool[NB_THREADS];

    void visitVertices(uint source_node);
    void visitVerticesThreadWork(uint source_node);
    void updateOutputThreadWork(uint vertex, uint neighboor, uint8_t edgeWeight);
    void refillVertexQueue(uint source_node);
};

#endif // CPU_SOLVER_CPP
