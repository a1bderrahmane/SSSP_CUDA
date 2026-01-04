#ifndef HYBRID_SOLVER_CUH
#define HYBRID_SOLVER_CUH

#include "Isolver.cuh"
#include "CSR.hpp"
#include "cuda_runtime.h"
#include "utils.cuh"
#include "ConcurrentQueue.cpp"

#include <mutex>
#include <thread>

static constexpr int BLOCK_SIZE = 1024;
static constexpr uint8_t GRID_SIZE = 40;
static constexpr uint8_t WARP_SIZE = 32;

static constexpr int NB_CPU_THREADS = 8;

class HybridSolver : public Isolver {

public:
    std::vector<uint> solve(uint source_node) override;
    HybridSolver(const std::string &filename);
    ~HybridSolver() override;

private:
    CSR *csr_graph;

    // Unified memory data
    uint *row_ptr;
    uint *col_idx;
    uint8_t *weights;
    int* nbVerticesPointer;
    int* nbEdgesPointer;
    uint *distances;
    uint *predecessors;

    // Host data
    bool* hostVerticesUpdated;
    std::mutex hostOutputWriterMutex;
    ConcurrentQueue *hostVertexQueue;
    std::thread* hostThreadPool[NB_CPU_THREADS];
    void HybridSolver::hostKernel();
    void HybridSolver::hostUpdateOutput(uint vertex, uint neighboor, uint8_t edgeWeight);

    // Device data
    // TODO

    void allocateMemory();
    void initializeData();
    __global__ void deviceKernel(
        int num_vertices,
        const uint *row_ptr,
        const uint *col_idx,
        const uint8_t *weights,
        uint *distances,
        const uint *workFront_in,
        uint *workFront_out
    );
};

#endif // HYBRID_SOLVER_CUH