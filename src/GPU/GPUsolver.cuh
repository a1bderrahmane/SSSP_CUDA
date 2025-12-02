#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include "Isolver.cuh"
#include "CSR.hpp"
#include "cuda_runtime.h"
#include "utils.cuh"

static constexpr uint8_t BLOCK_size = 1024;
static constexpr uint8_t GRID_SIZE = 40;
static constexpr uint8_t WARP_SIZE = 32;
class GPUsolver : public Isolver
{
public:
    std::vector<int> solve(const std::string &filename, int source_node) override;
    GPUsolver(const std::string &filename);
    ~GPUsolver() override;

protected:
    __device__ void traverseGraph(int source_node, __shared__ uint8_t *local_near_pile);
    __global__ void solver_kernel(int source_node);

private:
    CSR *csr_graph;
    uint *row_ptr;
    uint *col_idx;
    u_int8_t *weights;
    uint *col_idx_device;
    uint *row_ptr_device;
    uint *distances;
    uint *distances_device;
    uint8_t *weights_device;
    uint *nearPile;
    uint *farPile;
    int *delta;
    int *nearPileSize;
    int *farPileSize;
    void allocate_device_memory();
    void transfer_data_from_host_to_device();
    void transfer_data_from_device_to_host();
};

#endif // GPU_SOLVER_CUH