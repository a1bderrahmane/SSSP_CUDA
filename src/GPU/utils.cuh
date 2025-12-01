#ifndef UTILS_CUH
#define UTILS_CUH
#include "CSR.hpp"
#include "GPUsolver.cuh"
typedef std::pair<std::pair<uint, uint>, std::pair<uint, uint>> CSREdgesRange;
typedef std::pair<uint, uint> CSREdge;
class utils
{
public:
    __device__ __forceinline__ int get_global_id()
    {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ __forceinline__ int get_total_threads()
    {
        return gridDim.x * blockDim.x;
    }

    // Binary search will be used in order to find edge source in the CSR file in O (log(nb edges))
    __device__ uint binary_search(uint *arr, uint target, int size)
    {
        int low = 0;
        int high = size;
        int result = size;
        while (low < high)
        {
            int mid = low + (high - low) / 2;

            if (arr[mid] > target)
            {
                result = mid;
                high = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        return result;
    }

    // method to find an edge in the CSR using the binary search
    __device__ CSREdge findCSREdge(CSR *csr_graph, int edgeIdx)
    {
        int source = binary_search(csr_graph->getRowPtr(), edgeIdx, csr_graph->getNumberOfEdges() + 1) - 1;
        if (source >= edgeIdx)
        {
            std::cerr << "Check Error here" << std::endl;
        }
        int destination = csr_graph->getColIdx()[edgeIdx];
        CSREdge edge = {source, destination};
    }
    __device__ CSREdgesRange getBlockEdgesRange(CSR *csr_graph)
    {

        int num_edges = csr_graph->getNumberOfEdges();
        int block_start_edge = floor((blockIdx.x * num_edges) / BLOCK_size);
        int block_end_edge = floor(((blockIdx.x + 1) * num_edges) / BLOCK_size);
    }
    __device__ CSREdgesRange getMyEdgesRange(CSR *csr_graph)
    {
    }

private:
};
#endif