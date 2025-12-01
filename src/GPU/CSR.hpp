#ifndef CSR_HPP
#define CSR_HPP
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <unordered_map>

#define maxWeight 50;
#define minWeight 5;

struct Edge
{
    uint source;
    uint destination;
};
// We represent the graph as a Compact Sparse Row data structure
// It consists of three arrays intending to make the sparse graph be reprensented using less memory
// This is gonna help a lot with the data transfer from host to device and vice versa
class CSR
{
public:
    CSR(const std::string &filename);
    ~CSR();

    std::vector<std::pair<uint, uint>> edges;
    int num_vertices;
    int num_edges;
    uint *row_ptr;
    uint *col_idx;
    u_int8_t *weights;
    std::unordered_map<int, std::vector<int>> adjacencies;
    int getNumberOfEdges();
    int getNumberofVertices();
    uint *getRowPtr();
    uint *getColIdx();
    u_int8_t *getWeights();

private:
    u_int8_t generateRandomWeight();
    void allocateMemory();
    void makeAdjacencies(const std::string &filename);
    void makeRowPtr();
    void makeColIdx();
};
#endif
