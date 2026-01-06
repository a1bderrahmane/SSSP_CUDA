#include "CPUSolver.hpp"

CPUSolver::CPUSolver(const std::string &filename) {
    csr_graph = new CSR(filename);
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

std::vector<uint> CPUSolver::solve(uint source_node) {
    distances[source_node] = 0;
    vertexQueue = new ConcurrentQueue();
    vertexQueue->enqueue(source_node);

    while (!vertexQueue->isEmpty()) {
        visitVertices(source_node);
        refillVertexQueue(source_node);
    }

    return {};
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
            uint8_t edgeWeight = weights[i];
            
            // Take the lock only if it seems that we need to update output
            if (distances[neighboor] > distances[vertex] + edgeWeight) {
                updateOutputThreadWork(vertex, neighboor, edgeWeight);
            }
        }
    }
}

void CPUSolver::updateOutputThreadWork(uint vertex, uint neighboor, uint8_t edgeWeight) {
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

void CPUSolver::printResults(std::ostream &out) {
    out << "### CPU Solver : Results ###\n";
    for (uint vertex = 0; vertex < csr_graph->getNumberofVertices(); vertex++) {
        out << "- distance to vertex " << vertex << " : " << distances[vertex] << "\n";
        out << "  predecessor of " << vertex << " : " << predecessors[vertex] << "\n";
    }
}

CPUSolver::~CPUSolver() {
    free(distances);
    free(predecessors);
}
