#ifndef HYBRID_SOLVER_CUH
#define HYBRID_SOLVER_CUH

#include "Isolver.hpp"
#include "CSR.hpp"
#include "ConcurrentQueue.hpp"

#include <mutex>
#include <thread>

static constexpr uint8_t HYBRID_TPB = 32;

static constexpr int NB_CPU_THREADS = 2;

class HybridSolver : public Isolver {

public:
    std::vector<uint> solve(uint source_node) override;
    void printDistances();
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
    bool* verticesUpdated;
    bool* deviceVertexQueue;

    // Host
    std::mutex hostOutputWriterMutex;
    ConcurrentQueue *hostVertexQueue;
    std::thread* hostThreadPool[NB_CPU_THREADS];
    void hostKernelLaunch();
    void hostKernel();
    void hostUpdateOutput(uint vertex, uint neighboor, uint8_t edgeWeight);
    void refillHostVertexQueue();

    // Device
    void refillDeviceVertexQueue();
    void deviceKernelLaunch(uint nbVertices);

    // Main thread methods
    void allocateMemory();
    void initializeData();
    uint countVerticesInQueue();
};

#endif // HYBRID_SOLVER_CUH