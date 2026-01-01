#include "CPUSolver.hpp"

CPUSolver::CPUSolver(const std::string &filename) {
    csr_graph = new CSRGraph(filename);
    row_ptr = csr_graph->getRowPtr();
    col_idx = csr_graph->getColIdx();
    weights = csr_graph->getWeights();

    distances = (uint*) malloc(csr_graph->getNumberofVertices() * sizeof(uint));
    predecessors = (uint*) malloc(csr_graph->getNumberofVertices() * sizeof(uint));
    verticesUpdated = (bool*) malloc(csr_graph->getNumberofVertices() * sizeof(bool));

    for (int i = 0; i < csr_graph->getNumberofVertices(); i++) {
        distances[i] = UINT_INFINITY;
        verticesUpdated[i] = false;
    }
}

void CPUSolver::solve(uint source_node) {
    vertexQueue = new ConcurrentQueue();
    vertexQueue->enqueue(source_node);

    while (!vertexQueue->isEmpty()) {
        printf("[solve] new round of visiting vertices\n");
        visitVertices(source_node);
        refillVertexQueue(source_node);
    }
}

void CPUSolver::visitVertices(uint source_node) {
    // launch all threads
    for (int i = 0; i < NB_THREADS; i++) {
        threadPool[i] = new std::thread (&CPUSolver::visitVerticesThreadWork, this, source_node);
    }

    // wait for all threads to finish
    for (int i = 0; i < NB_THREADS; i++) {
        threadPool[i]->join();
        delete threadPool[i];
    }
}

void CPUSolver::visitVerticesThreadWork(uint source_node) {
    // Dequeue vertices until vertex queue is empty
    while (true) {
        uint vertex = vertexQueue->dequeue();

        if (vertex == UINT_INFINITY) {
            // Queue is empty
            return;
        }
        
        // Iterate on every edges of our vertex
        for (int i = row_ptr[vertex]; i < row_ptr[vertex + 1]; i++) {
            uint neighboor = col_idx[i];
            uint edgeWeight = weights[i];
            
            // Take the lock only if it seems that we need to update output
            if (distances[neighboor] > distances[vertex] + edgeWeight) {
                updateOutputThreadWork(vertex, neighboor, edgeWeight);
            }
        }
    }
}

void CPUSolver::updateOutputThreadWork(uint vertex, uint neighboor, uint edgeWeight) {
    std::lock_guard<std::mutex> lock(outputWriterMutex);

    // Ensure with the lock that we need to update output
    if (distances[neighboor] > distances[vertex] + edgeWeight) {
        distances[neighboor] = distances[vertex] + edgeWeight;
        predecessors[neighboor] = vertex;
        verticesUpdated[neighboor] = true;
    }
}

void CPUSolver::refillVertexQueue(uint source_node) {
    vertexQueue = new ConcurrentQueue();

    for (int vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        if (verticesUpdated[vertex]) {
            vertexQueue->enqueue(vertex);
            verticesUpdated[vertex] = false;
        }
    }
}

void CPUSolver::printResults() {
    printf("### CPU Solver : Results ###\n");
    for (uint vertex = 0; vertex < csr_graph->getNumberOfEdges(); vertex++) {
        printf("- distance to vertex %u : %u\n", vertex, distances[vertex]);
    }
}

CPUSolver::~CPUSolver() {
    free(distances);
    free(predecessors);
}
