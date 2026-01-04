#include "HybridSolver.cuh"


HybridSolver::HybridSolver(const std::string &filename) {
    csr_graph = new CSR(filename);

    allocateMemory();
    initializeData();
}

void HybridSolver::allocateMemory() {
    cudaMallocManaged(&row_ptr, (csr_graph->getNumberofVertices() + 1) * sizeof(uint));
    cudaMallocManaged(&col_idx, csr_graph->getNumberOfEdges() * sizeof(uint));
    cudaMallocManaged(&weights, csr_graph->getNumberOfEdges() * sizeof(uint));

    cudaMallocManaged(&nbEdgesPointer, sizeof(int));
    cudaMallocManaged(&nbVerticesPointer, sizeof(int));

    cudaMallocManaged(&distances, csr_graph->getNumberofVertices() * sizeof(uint));
    cudaMallocManaged(&predecessors, csr_graph->getNumberofVertices() * sizeof(uint));

    // Host memory
    hostVerticesUpdated = (bool*) malloc(csr_graph->getNumberofVertices() * sizeof(bool));

    // Device memory
    // TODO (if needed)
}

void HybridSolver::initializeData() {
    memcpy(row_ptr, csr_graph->getRowPtr(), (csr_graph->getNumberofVertices() + 1) * sizeof(uint));
    memcpy(col_idx, csr_graph->getColIdx(), csr_graph->getNumberOfEdges() * sizeof(uint));
    memcpy(col_idx, csr_graph->getWeights(), csr_graph->getNumberOfEdges() * sizeof(uint));

    *nbVerticesPointer = csr_graph->getNumberofVertices();
    *nbEdgesPointer = csr_graph->getNumberOfEdges();

    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        distances[vertex] = UINT_INFINITY;
        predecessors[vertex] = UINT_INFINITY;
        hostVerticesUpdated[vertex] = false;
    }
}

std::vector<uint> HybridSolver::solve(uint source_node) {
    distances[source_node] = 0;
    hostVertexQueue = new ConcurrentQueue();
    hostVertexQueue->enqueue(source_node);

    // Need to update this value after each iteration
    uint nbNodesInQueue = 1;

    while (nbNodesInQueue > 0) {
        if (nbNodesInQueue < NB_CPU_THREADS) {
            // CPU iteration
        } else {
            // GPU iteration
        }
    }

    return {};
}


//////////////////////////////////////////////////////////////////////////////////////////
///                                 Host kernel                                        ///
//////////////////////////////////////////////////////////////////////////////////////////


void HybridSolver::hostKernel() {
    // Dequeue vertices until vertex queue is empty
    while (true) {
        uint vertex = hostVertexQueue->dequeue();

        if (vertex == UINT_INFINITY) {
            // Queue is empty
            return;
        }
        
        // Iterate on every edges of our vertex
        for (int i = row_ptr[vertex]; i < row_ptr[vertex + 1]; i++) {
            uint neighboor = col_idx[i];
            uint8_t edgeWeight = weights[i];
            
            // Take the lock only if it seems that we need to update output
            if (distances[neighboor] > distances[vertex] + edgeWeight) {
                hostUpdateOutput(vertex, neighboor, edgeWeight);
            }
        }
    }
}

void HybridSolver::hostUpdateOutput(uint vertex, uint neighboor, uint8_t edgeWeight) {
    std::lock_guard<std::mutex> lock(hostOutputWriterMutex);

    // Ensure with the lock that we need to update output
    if (distances[neighboor] > distances[vertex] + edgeWeight) {
        distances[neighboor] = distances[vertex] + edgeWeight;
        predecessors[neighboor] = vertex;
        hostVerticesUpdated[neighboor] = true;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
///                                 Device kernel                                      ///
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void HybridSolver::deviceKernel(
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

    for (int i = tid; i < num_vertices; i += totalThreads) {
        if (workFront_in[i] == 1) {
            uint start_edge = row_ptr[i];
            uint end_edge = row_ptr[i + 1];

            for (uint edge = start_edge; edge < end_edge; edge++) {
                int destination = col_idx[edge];
                int weight = weights[edge];
                
                if(distances[i] == UINT_MAX) continue;

                uint new_dist = distances[i] + weight;
                
                uint old_dist = atomicMin(&distances[destination], new_dist);

                if (new_dist < old_dist) {
                    workFront_out[destination] = 1;
                }
            }
        }
    }
}