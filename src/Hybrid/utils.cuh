#ifndef UTILS_CUH
#define UTILS_CUH
#include "CSR.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace utils
{
    int inline computeDelta(int averageDegree, int averageEdgeWeight)
    {
        int WARP_SIZE=32;
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
    struct IsNonZero {
            __host__ __device__ bool operator()(int x) const { return x != 0; }
        };
    inline void deduplicate(thrust::device_vector<uint> &arr)
    {
        if (arr.empty())
        {
            return;
        }
        thrust::sort(arr.begin(), arr.end());
        auto new_end = thrust::unique(arr.begin(), arr.end());
        arr.erase(new_end, arr.end());
    }
}
#endif