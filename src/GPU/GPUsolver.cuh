#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include "Isolver.cuh"
#include "CSR.hpp"
#include "cuda_runtime.h"

#define BLOCK_size 1024
#define GRID_SIZE 40
class GPUsolver : public Isolver
{
public:
    std::vector<int> solve(const std::string &filename, int source_node) override;
    GPUsolver(const std::string &filename);
    ~GPUsolver() override;

private:
    CSR *csr_graph;
    uint *row_ptr;
    uint *col_idx;
    u_int8_t *weights;
    uint*col_idx_device;
    uint*row_ptr_device;
    uint8_t*weights_device;
    void allocate_device_memory();
    void allocate_host_memory();
    void transfer_data_from_host_to_device();
    void transfer_data_from_device_to_host();
};

#endif // GPU_SOLVER_CUH