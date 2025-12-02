#ifndef UTILS_CUH
#define UTILS_CUH
#include "CSR.hpp"
#include "GPUsolver.cuh"
#include <cub/cub.cuh>

typedef std::pair<uint, uint> CSREdge;
typedef std::pair<CSREdge, CSREdge> CSREdgesRange;
namespace utils
{
    int inline computeDelta(int averageDegree, int averageEdgeWeight)
    {
        return (WARP_SIZE * averageEdgeWeight) / averageDegree;
    }
    __device__ __forceinline__ int get_global_id()
    {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ __forceinline__ int get_total_threads()
    {
        return gridDim.x * blockDim.x;
    }
    __device__ inline uint xorshift32(uint &state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    __device__ int random_threadIdx()
    {
        uint seed = clock64() ^ (threadIdx.x + blockIdx.x * 12345);
        uint rnd = xorshift32(seed);
        return rnd & (BLOCK_size - 1);
    }
    // Binary search will be used in order to find edge source in the CSR file in O (log(nb edges))
    __device__ uint binary_search(uint *arr, uint target, int size, int start, int end)
    {
        int low = start;
        int high = end;
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
    __device__ CSREdge findCSREdge(CSR *csr_graph, int edgeIdx, int start, int end)
    {
        int source = binary_search(csr_graph->getRowPtr(), edgeIdx, csr_graph->getNumberOfEdges() + 1, start, end) - 1;
        if (source >= edgeIdx)
        {
            std::cerr << "Check Error here" << std::endl;
        }
        int destination = csr_graph->getColIdx()[edgeIdx];
        CSREdge edge = {source, destination};
    }
    __device__ void deduplicateArray(uint *arr, int arrLength)
    {
    }
    __device__ void getBlockEdgesRange(CSR *csr_graph, __shared__ CSREdgesRange *range_csr)
    {
        int num_edges = csr_graph->getNumberOfEdges();
        CSREdge block_start_edge_csr;
        CSREdge block_end_edge_csr;
        int rnd1 = random_threadIdx();
        if (threadIdx.x == rnd1)
        {
            int block_start_edge = floor((blockIdx.x * num_edges) / GRID_SIZE);
            CSREdge block_start_edge_csr = findCSREdge(csr_graph, block_start_edge, 0, csr_graph->getNumberOfEdges() + 1);
        }
        int rnd2 = random_threadIdx();
        if (threadIdx.x == rnd2)
        {
            int block_end_edge = floor(((blockIdx.x + 1) * num_edges) / GRID_SIZE);
            CSREdge block_end_edge_csr = findCSREdge(csr_graph, block_end_edge, 0, csr_graph->getNumberOfEdges() + 1);
        }
        __syncthreads();
        *range_csr = {block_start_edge_csr, block_end_edge_csr};
    }
    __device__ CSREdgesRange getMyEdgesRange(CSR *csr_graph, __shared__ CSREdgesRange *range_csr)
    {
        int t = threadIdx.x;
        int block_start_edge = floor((blockIdx.x * csr_graph->getNumberOfEdges()) / GRID_SIZE);
        int block_end_edge = floor(((blockIdx.x + 1) * csr_graph->getNumberOfEdges()) / GRID_SIZE);
        int R = block_end_edge - block_start_edge;
        int thread_edge_start = block_start_edge + floor(t * R / BLOCK_size);
        int thread_edge_end = block_start_edge + floor((t + 1) * R / BLOCK_size);
        CSREdge thread_edge_start_csr = findCSREdge(csr_graph, thread_edge_start, block_start_edge, block_end_edge + 1 + 1);
        CSREdge thread_edge_end_csr = findCSREdge(csr_graph, thread_edge_end, thread_edge_start_csr.first, block_end_edge + 1 + 1);
        return {thread_edge_start_csr, thread_edge_end_csr};
    }

};
#endif