#include "CSR.hpp"

CSR::CSR(const std::string &filename)
{
    makeAdjacencies(filename);
    makeRowPtr();
    makeColIdx();
}

uint CSR::generateRandomWeight()
{
    std::random_device dev;
    std::mt19937 rng(dev());
    int a =minWeight;
    int b =maxWeight;
    std::uniform_int_distribution<std::mt19937::result_type> gen(a,b);
    return (uint)gen(rng);
}

void CSR::makeAdjacencies(const std::string&filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    uint u, v;
    uint max_index = 0;
    while (file >> u >> v)
    {
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

void CSR::makeRowPtr()
{
    row_ptr.reserve(num_vertices + 1);
    int cumulativeCount = 0;
    for (uint i = 0; i < num_vertices; i++)
    {
        row_ptr.push_back(cumulativeCount);
        if (adjacencies.find(i) != adjacencies.end())
        {
            cumulativeCount += adjacencies[i].size();
        }
    }
    row_ptr.push_back(cumulativeCount);
}

void CSR::makeColIdx()
{
    for(uint i =0;i<num_vertices;i++)
    {
        if (adjacencies.find(i)!=adjacencies.end())
        {
            for(auto&v:adjacencies[i])
            {
                col_idx.push_back(v);
                weights.push_back(generateRandomWeight());
            }
        }
    }
}
int main()
{
    std::string file = "/home/a1bderrahmane/SSSP_CUDA/datasets/simple-graph.txt";
    return 0;
}
