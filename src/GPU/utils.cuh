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
    void deduplicate(thrust::device_vector<uint> &arr)
    {
        if (arr.empty())
        {
            return;
        }
        thrust::sort(arr.begin(), arr.end());
        auto new_end = thrust::unique(arr.begin(), arr.end());
        arr.erase(new_end, arr.end());
    }
    struct IsNonZero
    {
        __host__ __device__ bool operator()(int x) const
        {
            return x != 0;
        }
    };
    thrust::device_vector<int> compact(thrust::device_vector<uint> &arr)
    {
        if (arr.empty())
        {
            return thrust::device_vector<int>();
        }
        thrust::device_vector<int> d_output(arr.size());
        auto d_indices_begin = thrust::make_counting_iterator(0);
        auto new_end = thrust::copy_if(
            thrust::cuda::par,
            d_indices_begin,
            d_indices_begin + arr.size(),
            arr.begin(), 
            d_output.begin(),
            IsNonZero() 
        );
        d_output.resize(std::distance(d_output.begin(), new_end));
        return d_output;
    }
    struct IsNonZero {
        __host__ __device__ bool operator()(int x) const { return x != 0; }
    };
};
#endif