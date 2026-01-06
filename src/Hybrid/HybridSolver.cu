#include "HybridSolver.cuh"


HybridSolver::HybridSolver(const std::string &filename) {
    csr_graph = new CSR(filename);
    printf("number of vertices in the graph : %d\n", csr_graph->getNumberofVertices());
    printf("allocating unified memory...\n");
    allocateMemory();
    printf("initializing data...\n");
    initializeData();
}

HybridSolver::~HybridSolver() {
    cudaFree(row_ptr);
    cudaFree(col_idx);
    cudaFree(weights);

    cudaFree(nbEdgesPointer);
    cudaFree(nbVerticesPointer);

    cudaFree(distances);
    cudaFree(predecessors);
    cudaFree(verticesUpdated);
    cudaFree(deviceVertexQueue);
}

void HybridSolver::allocateMemory() {
    cudaMallocManaged(&row_ptr, (csr_graph->getNumberofVertices() + 1) * sizeof(uint));
    cudaMallocManaged(&col_idx, csr_graph->getNumberOfEdges() * sizeof(uint));
    cudaMallocManaged(&weights, csr_graph->getNumberOfEdges() * sizeof(uint));

    cudaMallocManaged(&nbEdgesPointer, sizeof(int));
    cudaMallocManaged(&nbVerticesPointer, sizeof(int));

    cudaMallocManaged(&distances, csr_graph->getNumberofVertices() * sizeof(uint));
    cudaMallocManaged(&predecessors, csr_graph->getNumberofVertices() * sizeof(uint));
    cudaMallocManaged(&verticesUpdated, csr_graph->getNumberofVertices() * sizeof(bool));
    cudaMallocManaged(&deviceVertexQueue, csr_graph->getNumberofVertices() * sizeof(bool));
}

void HybridSolver::initializeData() {
    printf("  starting memcpy...\n");
    memcpy(row_ptr, csr_graph->getRowPtr(), (csr_graph->getNumberofVertices() + 1) * sizeof(uint));
    memcpy(col_idx, csr_graph->getColIdx(), csr_graph->getNumberOfEdges() * sizeof(uint));
    memcpy(weights, csr_graph->getWeights(), csr_graph->getNumberOfEdges() * sizeof(uint));

    printf("  starting write on pointers...\n");
    *nbVerticesPointer = csr_graph->getNumberofVertices();
    *nbEdgesPointer = csr_graph->getNumberOfEdges();

    printf("  starting loop initialization...\n");
    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        distances[vertex] = UINT_INFINITY;
        predecessors[vertex] = UINT_INFINITY;
        verticesUpdated[vertex] = false;
        deviceVertexQueue[vertex] = false;
    }
}

std::vector<uint> HybridSolver::solve(uint source_node) {
    printf("[Hybrid] Starting to solve with source node %u...\n", source_node);
    distances[source_node] = 0;

    // Mark source_node as updated so that it is added to the first vertexQueue
    verticesUpdated[source_node] = true;

    uint nbVerticesInQueue = 1;

    while (nbVerticesInQueue > 0) {
        printDistances();
        if (nbVerticesInQueue < NB_CPU_THREADS) {
            printf("[Hybrid] new host iteration...\n");
            refillHostVertexQueue();
            hostKernelLaunch();
        } else {
            printf("[Hybrid] new device iteration...\n");
            refillDeviceVertexQueue();
            deviceKernelLaunch(nbVerticesInQueue);
        }

        nbVerticesInQueue = countVerticesInQueue();
    }

    return {};
}

void HybridSolver::printDistances() {
    printf("### Hybrid Solver : Results ###\n");
    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        printf("- distance to vertex %d : %d\n", vertex, distances[vertex]);
    }
}

uint HybridSolver::countVerticesInQueue() {
    uint nbVertices = 0;
    printf("counting vertices in queue...\n");
    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        if (verticesUpdated[vertex]) {
            printf("  %d is in queue\n", vertex);
            nbVertices++;
        }
    }
    
    return nbVertices;
}


//////////////////////////////////////////////////////////////////////////////////////////
///                                 Host kernel                                        ///
//////////////////////////////////////////////////////////////////////////////////////////


void HybridSolver::hostKernelLaunch() {
    // launch all threads
    for (int i = 0; i < NB_CPU_THREADS; i++) {
        hostThreadPool[i] = new std::thread (&HybridSolver::hostKernel, this);
    }

    // wait for all threads to finish
    for (int i = 0; i < NB_CPU_THREADS; i++) {
        hostThreadPool[i]->join();
        delete hostThreadPool[i];
    }
}

void HybridSolver::hostKernel() {
    // Dequeue vertices until vertex queue is empty
    while (true) {
        uint vertex = hostVertexQueue->dequeue();

        if (vertex == UINT_INFINITY) {
            // Queue is empty
            return;
        }
        
        // Iterate on every edges of our vertex
        for (uint i = row_ptr[vertex]; i < row_ptr[vertex + 1]; i++) {
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
        verticesUpdated[neighboor] = true;
    }
}

void HybridSolver::refillHostVertexQueue() {
    hostVertexQueue = new ConcurrentQueue();

    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        if (verticesUpdated[vertex]) {
            hostVertexQueue->enqueue(vertex);
            verticesUpdated[vertex] = false;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
///                                 Device kernel                                      ///
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void deviceKernel(
    bool* deviceVertexQueue,
    bool* verticesUpdated,
    uint* row_ptr,
    uint* col_idx,
    uint8_t* weights,
    uint* distances,
    uint nbVerticesInGraph
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = tid; i < nbVerticesInGraph; i += totalThreads) {
        if (deviceVertexQueue[i]) {
            uint start_edge = row_ptr[i];
            uint end_edge = row_ptr[i + 1];

            for (uint edge = start_edge; edge < end_edge; edge++) {
                uint destination = col_idx[edge];
                uint8_t weight = weights[edge];
                
                if(distances[i] == UINT_INFINITY) continue;

                uint new_dist = distances[i] + weight;
                
                uint old_dist = atomicMin(&distances[destination], new_dist);

                if (new_dist < old_dist) {
                    verticesUpdated[destination] = true;
                }
            }
        }
    }
}

void HybridSolver::refillDeviceVertexQueue() {
    // set new device vertex queue to old verticesUpdated
    memcpy(deviceVertexQueue, verticesUpdated, csr_graph->getNumberofVertices() * sizeof(bool));

    // reset verticesUpdated
    memset(verticesUpdated, (int) false, (size_t) csr_graph->getNumberofVertices());
}

void HybridSolver::deviceKernelLaunch(uint nbVertices) {
    int numBlocks = (nbVertices + HYBRID_TPB - 1) / HYBRID_TPB;
        // if (numBlocks > 1024) numBlocks = 1024;
        deviceKernel<<<numBlocks, HYBRID_TPB>>>(
            deviceVertexQueue,
            verticesUpdated,
            row_ptr,
            col_idx,
            weights,
            distances,
            nbVertices
        );
        cudaDeviceSynchronize();
}
