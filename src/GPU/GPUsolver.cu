#include "GPUsolver.cuh"

GPUsolver::GPUsolver(const std::string &filename)
{
    csr_graph = new CSR(filename);
    col_idx = csr_graph->getColIdx();
    row_ptr = csr_graph->getRowPtr();
    weights = csr_graph->getWeights();
}


void GPUsolver::allocate_device_memory()
{
    
    cudaMalloc(&col_idx_device,sizeof(col_idx));

    cudaMalloc(&row_ptr_device,sizeof(row_ptr));
    cudaMalloc(&weights_device,sizeof(weights_device));
}

void GPUsolver::transfer_data_from_host_to_device()
{
    cudaMemcpy(col_idx_device,col_idx,sizeof(col_idx),cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_device,row_ptr,sizeof(row_ptr),cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device,weights,sizeof(weights),cudaMemcpyHostToDevice);
}


