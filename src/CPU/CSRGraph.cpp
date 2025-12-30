#include "CSRGraph.hpp"

CSRGraph::CSRGraph(const std::string &filename) {
    num_edges = 0;
    makeAdjacencies(filename);
    allocateMemory();
    makeRowPtr();
    makeColIdx();
}

void CSRGraph::makeAdjacencies(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    uint u, v;
    uint max_index = 0;
    while (file >> u >> v) {
        num_edges++;
        adjacencies[u].push_back(v);
        // Track the largest vertex index found
        if (u > max_index)
            max_index = u;
        if (v > max_index)
            max_index = v;
    }

    // The number of vertices is the maximum index + 1 (since indices are 0-based)
    this->num_vertices = max_index + 1;
}

void CSRGraph::allocateMemory() {
    row_ptr = (uint*) malloc((num_vertices + 1) * sizeof(uint));
    size_t col_idx_size = 0;

    for (uint i = 0; i < num_vertices; i++) {
        if (adjacencies.find(i) != adjacencies.end()) {
            col_idx_size += adjacencies[i].size();
        }
    }

    col_idx = (uint*) malloc(sizeof(uint)*col_idx_size);
    weights = (uint*) malloc(sizeof(u_int8_t)*col_idx_size);
}

void CSRGraph::makeRowPtr() {
    int cumulativeCount = 0;
    for (uint i = 0; i < num_vertices; i++) {
        row_ptr[i] = cumulativeCount;
        if (adjacencies.find(i) != adjacencies.end()) {
            cumulativeCount += adjacencies[i].size();
        }
    }

    row_ptr[num_vertices] = cumulativeCount;
}

void CSRGraph::makeColIdx() {
    int j=0;
    for (uint i = 0; i < num_vertices; i++) {
        if (adjacencies.find(i) != adjacencies.end()) {
            for (auto &v : adjacencies[i]) {
                col_idx[j]=v;
                weights[j]=(u_int8_t)generateRandomWeight();
                j++;
            }
        }
    }
}

uint CSRGraph::generateRandomWeight() {
    return 1;
}

int CSRGraph::getNumberOfEdges() {
    return num_edges;
}

int CSRGraph::getNumberofVertices() {
    return num_vertices;
}

uint* CSRGraph::getColIdx() {
    return col_idx;
}

uint* CSRGraph::getRowPtr() {
    return row_ptr;
}

uint* CSRGraph::getWeights() {
    return weights;
}

int CSRGraph::getAverageDegree() {
    
    int degreeCumulativeSum = row_ptr[num_edges];
    return degreeCumulativeSum / num_edges;
}

int CSRGraph::getAverageEdgeWeight() {
    return 1;
}

CSRGraph::~CSRGraph() {
    free(row_ptr);
    free(col_idx);
    free(weights);
}
