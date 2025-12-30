#ifndef CSR_GRAPH_HPP
#define CSR_GRAPH_HPP
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <unordered_map>


// We represent the graph as a Compact Sparse Row data structure
// It consists of three arrays intending to make the sparse graph be reprensented using less memory
// This is gonna help a lot with the data transfer from host to device and vice versa
class CSRGraph
{
public:
    CSRGraph(const std::string &filename);
    ~CSRGraph();

    int getNumberOfEdges();
    int getNumberofVertices();
    uint *getRowPtr();
    uint *getColIdx();
    uint *getWeights();
    int getAverageDegree();
    int getAverageEdgeWeight();
    

private:
    std::vector<std::pair<uint, uint>> edges;
    int num_vertices;
    int num_edges;
    uint *row_ptr;
    uint *col_idx;
    uint *weights;
    std::unordered_map<int, std::vector<int>> adjacencies;

    uint generateRandomWeight();
    void allocateMemory();
    void makeAdjacencies(const std::string &filename);
    void makeRowPtr();
    void makeColIdx();
};
#endif
