// #ifndef GPU_SOLVER_CUH
// #define GPU_SOLVER_CUH

// #include "Isolver.hpp"
// #include "CSR.hpp"
// #include <thrust/fill.h>
// #include <thrust/device_vector.h>

// static constexpr int BLOCK_SIZE = 1024;
// static constexpr uint8_t GRID_SIZE = 40;
// static constexpr uint8_t WARP_SIZE = 32;
// class GPUsolver : public Isolver
// {
// public:
//     std::vector<uint> solve(uint source_node) override;
//     GPUsolver(const std::string &filename);
//     ~GPUsolver() override;
//     std::vector<uint> getDistancesHost();

// protected:
//      void workFrontSweepSolver(int source_node);

// private:
// CSR *csr_graph;
//    uint *d_row_ptr;
//     uint *d_col_idx;
//     uint8_t *d_weights;
//     int nbVertices;
//     int nbEdges;
//     thrust::device_vector<unsigned int> d_distances;
//     void allocate_device_memory();
//     void transfer_data_from_host_to_device();
//     void transfer_data_from_device_to_host();
    
// };

// #endif // GPU_SOLVER_CUH