#include "utils.cuh"

__global__ void k_workFrontSweep(
    int num_vertices,
    const uint *row_ptr, 
    const uint *col_idx, 
    const uint8_t *weights, 
    uint *distances, 
    const uint *workFront_in, 
    uint *workFront_out)
{
    int tid = utils::get_global_id();
    int totalThreads = utils::get_total_threads();

    for (int i = tid; i < num_vertices; i += totalThreads)
    {
        if (workFront_in[i] == 1)
        {
            uint start_edge = row_ptr[i];
            uint end_edge = row_ptr[i + 1];

            for (uint edge = start_edge; edge < end_edge; edge++)
            {
                int destination = col_idx[edge];
                int weight = weights[edge];
                
                if(distances[i] == UINT_MAX) continue; 

                uint new_dist = distances[i] + weight;
                
                uint old_dist = atomicMin(&distances[destination], new_dist);

                if (new_dist < old_dist)
                {
                    workFront_out[destination] = 1;
                }
            }
        }
    }
}