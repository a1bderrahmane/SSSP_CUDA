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
        distances[i] = INFINITY;
        verticesUpdated[i] = false;
    }
}

std::vector<int> CPUSolver::solve(const std::string &filename, uint source_node) {
    currentQueue = new ConcurrentQueue<uint>();
    currentQueue->enqueue(source_node);

    while (!currentQueue->isEmpty()) {
        nextQueue = new ConcurrentQueue<uint>();
    }
}

void CPUSolver::solveIteration(uint source_node) {
    
}

void CPUSolver::solveIterationThreadWork(uint source_node) {
    // Dequeue vertices until vertex queue is empty
    while (true) {
        uint vertex = currentQueue->dequeue();

        if (vertex == NULL) {
            // Queue is empty
            return;
        }
        
        // Iterate on every edges of our vertex
        for (int i = row_ptr[vertex]; i < row_ptr[vertex + 1]; i++) {
            uint neighboor = col_idx[i];
            uint edgeWeight = weights[i];
            
            // Take the lock only if it seems that we need to update output
            if (distances[vertex] > edgeWeight + distances[neighboor]) {
                updateOutputThreadWork(vertex, neighboor, edgeWeight + distances[neighboor]);
            }
        }
    }
}

void CPUSolver::updateOutputThreadWork(uint vertex, uint neighboor, uint newDistance) {
    std::lock_guard<std::mutex> lock(outputWriterMutex);

    // Ensure with the lock that we need to update output
    if (distances[vertex] > newDistance) {
        distances[vertex] = newDistance;
        predecessors[vertex] = neighboor;
        verticesUpdated[vertex] = true;
    }
}


CPUSolver::~CPUSolver() {
    free(distances);
    free(predecessors);
}
