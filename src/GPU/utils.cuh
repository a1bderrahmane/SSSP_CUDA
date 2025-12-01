#ifndef UTILS_CUH
#define UTILS_CUH
#include <stdio.h>
#include <cuda_runtime.h>
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

    __device__ uint binary_search(uint *arr, uint target, size_t size)
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

private:
};
#endif