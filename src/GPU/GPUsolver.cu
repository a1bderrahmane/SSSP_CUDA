#include "GPUsolver.cuh"


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
GPUsolver::GPUsolver(const std::string &filename)
{
    csr_graph = new CSR(filename);
    nbVertices = csr_graph->getNumberofVertices();
    nbEdges = csr_graph->getNumberOfEdges();

    allocate_device_memory();
    transfer_data_from_host_to_device();
}

void GPUsolver::allocate_device_memory()
{
    cudaMalloc(&d_row_ptr, sizeof(uint) * (nbVertices + 1));
    cudaMalloc(&d_col_idx, sizeof(uint) * nbEdges);
    cudaMalloc(&d_weights, sizeof(uint8_t) * nbEdges);

    d_distances.resize(nbVertices);
    thrust::fill(d_distances.begin(), d_distances.end(), UINT_MAX);
}

void GPUsolver::transfer_data_from_host_to_device()
{
    cudaMemcpy(d_row_ptr, csr_graph->getRowPtr(), sizeof(uint) * (nbVertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, csr_graph->getColIdx(), sizeof(uint) * nbEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, csr_graph->getWeights(), sizeof(uint8_t) * nbEdges, cudaMemcpyHostToDevice);
}


void GPUsolver::workFrontSweepSolver(int source_node)
{
    thrust::device_vector<uint> d_workFront(nbVertices, 0);
    thrust::device_vector<uint> d_workFront_next(nbVertices, 0);

    d_distances[source_node] = 0;
    d_workFront[source_node] = 1;
    int iter = 0;
    while (true)
    {
    
        uint active_nodes = thrust::reduce(d_workFront.begin(), d_workFront.end());
        if (active_nodes == 0) break;

        uint *ptr_dist = thrust::raw_pointer_cast(d_distances.data());
        uint *ptr_wf_in = thrust::raw_pointer_cast(d_workFront.data());
        uint *ptr_wf_out = thrust::raw_pointer_cast(d_workFront_next.data());

        int numBlocks = (nbVertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (numBlocks > 1024) numBlocks = 1024; 
        k_workFrontSweep<<<numBlocks, BLOCK_SIZE>>>(
            nbVertices,
            d_row_ptr,
            d_col_idx,
            d_weights,
            ptr_dist,
            ptr_wf_in,
            ptr_wf_out
        );
        cudaDeviceSynchronize();
        d_workFront = d_workFront_next;
        thrust::fill(d_workFront_next.begin(), d_workFront_next.end(), 0);
        
        iter++;
    }
    std::cout << "Converged in " << iter << " iterations." << std::endl;
}

std::vector<uint> GPUsolver::solve(uint source_node)
{
    workFrontSweepSolver(source_node);
    return {};

}
std::vector<uint> GPUsolver::getDistancesHost()
{
    std::vector<uint> h_dists(nbVertices);
    thrust::copy(d_distances.begin(), d_distances.end(), h_dists.begin());
    return h_dists;
}
GPUsolver::~GPUsolver()
{
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_weights);
    delete csr_graph;
}

