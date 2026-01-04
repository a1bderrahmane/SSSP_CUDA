// #ifndef GPU_SOLVER_CUH
// #define GPU_SOLVER_CUH

#include "Isolver.cuh"
#include "CSR.hpp"
#include "cuda_runtime.h"
#include "utils.cuh"
#include <thrust/fill.h>
#include <thrust/device_vector.h>
// #include <thrust/raw_pointer_cast.h>

static constexpr int BLOCK_SIZE = 1024;
static constexpr uint8_t GRID_SIZE = 40;
static constexpr uint8_t WARP_SIZE = 32;
class GPUsolver : public Isolver
{
public:
    std::vector<int> solve(uint source_node) override;
    GPUsolver(const std::string &filename);
    ~GPUsolver() override;
    std::vector<uint> getDistancesHost();

protected:
    // __global__ void workFrontSweep(uint*workFront,uint*workFront_output);
     void workFrontSweepSolver(int source_node);

private:
CSR *csr_graph;
   uint *d_row_ptr;
    uint *d_col_idx;
    uint8_t *d_weights;
    int nbVertices;
    int nbEdges;
    thrust::device_vector<unsigned int> d_distances;
    void allocate_device_memory();
    void transfer_data_from_host_to_device();
    void transfer_data_from_device_to_host();
    
};

// #endif // GPU_SOLVER_CUH