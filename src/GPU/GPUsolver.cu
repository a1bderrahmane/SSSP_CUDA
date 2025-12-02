#include "GPUsolver.cuh"

GPUsolver::GPUsolver(const std::string &filename)
{
    csr_graph = new CSR(filename);
    col_idx = csr_graph->getColIdx();
    row_ptr = csr_graph->getRowPtr();
    weights = csr_graph->getWeights();
    distances = csr_graph->getDistances();

}

void GPUsolver::allocate_device_memory()
{
    cudaMalloc(&col_idx_device, sizeof(col_idx));
    cudaMalloc(&row_ptr_device, sizeof(row_ptr));
    cudaMalloc(&weights_device, sizeof(weights_device));
    cudaMalloc(&distances_device, sizeof(distances));
    cudaMalloc(&nearPileSize,sizeof(int));
    cudaMalloc(&farPileSize,sizeof(int));
    cudaMalloc(&delta,sizeof(int));
    // Near far piles
    cudaMalloc(&nearPile, sizeof(uint) * csr_graph->getNumberofVertices());
    cudaMalloc(&farPile, sizeof(uint) * csr_graph->getNumberofVertices());
    cudaMemset(nearPile, 0, sizeof(uint) * csr_graph->getNumberofVertices());
    cudaMemset(farPile, 0, sizeof(uint) * csr_graph->getNumberofVertices());
    cudaMemset(nearPileSize,1,sizeof(int));
    cudaMemset(farPileSize,0,sizeof(int));
    cudaMemset(delta,utils::computeDelta(csr_graph->getAverageDegree(),csr_graph->getAverageEdgeWeight()),sizeof(int));
}

void GPUsolver::transfer_data_from_host_to_device()
{
    cudaMemcpy(col_idx_device, col_idx, sizeof(col_idx), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_device, row_ptr, sizeof(row_ptr), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device, weights, sizeof(weights), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_device, distances, sizeof(distances), cudaMemcpyHostToDevice);
}

void GPUsolver::transfer_data_from_device_to_host()
{
    cudaMemcpy(distances, distances_device, sizeof(distances), cudaMemcpyDeviceToHost);
}

__device__ void  GPUsolver::traverseGraph(int source_node, __shared__ uint8_t*local_near_pile)
{
    nearPile[0]=source_node;
    int i=0;
    while(*nearPileSize>0)
    {
           
    }
}

GPUsolver::~GPUsolver()
{
    cudaFree(col_idx_device);
    cudaFree(row_ptr_device);
    cudaFree(weights_device);
    cudaFree(distances_device);
    cudaFree(nearPile);
    cudaFree(farPile);
    cudaFree(nearPileSize);
    cudaFree(farPileSize);
    cudaFree(delta);
}

