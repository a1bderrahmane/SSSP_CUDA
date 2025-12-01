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
#define minWeight 5 ;


struct Edge
{
    uint source;
    uint destination;
};

class CSR
{
public:
    CSR(const std::string &filename);
    // ~CSR();
    
    std::vector<std::pair<uint, uint>> edges;
    int num_vertices;
    std::vector<int>row_ptr;
    std::vector<int>col_idx;
    std::vector<int>weights;
    std::unordered_map<int,std::vector<int>> adjacencies;

private:
    uint generateRandomWeight();
    void makeAdjacencies(const std::string&filename);
    void makeRowPtr();
    void makeColIdx();
};
#endif
