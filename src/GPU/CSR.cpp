#include "CSR.hpp"

CSR::CSR(const std::string &filename)
{
    num_edges=0;
    makeAdjacencies(filename);
    allocateMemory();
    makeRowPtr();
    makeColIdx();
}

u_int8_t CSR::generateRandomWeight()
{
    std::random_device dev;
    std::mt19937 rng(dev());
    u_int8_t a = minWeight;
    u_int8_t b = maxWeight;
    std::uniform_int_distribution<std::mt19937::result_type> gen(a, b);
    return (u_int8_t)gen(rng);
}

void CSR::makeAdjacencies(const std::string &filename)
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
void CSR::allocateMemory()
{
    row_ptr = (uint *)malloc((num_vertices + 1) * sizeof(uint));
    size_t col_idx_size = 0;
    for (uint i = 0; i < num_vertices; i++)
    {
        if (adjacencies.find(i) != adjacencies.end())
        {
            col_idx_size += adjacencies[i].size();
        }
    }
    col_idx=(uint*)malloc(sizeof(uint)*col_idx_size);
    weights=(u_int8_t *)malloc(sizeof(u_int8_t)*col_idx_size);

}
void CSR::makeRowPtr()
{
    int cumulativeCount = 0;
    for (uint i = 0; i < num_vertices; i++)
    {
        row_ptr[i] = cumulativeCount;
        if (adjacencies.find(i) != adjacencies.end())
        {
            cumulativeCount += adjacencies[i].size();
        }
    }
    row_ptr[num_vertices] = cumulativeCount;
}

void CSR::makeColIdx()
{
    int j=0;
    for (uint i = 0; i < num_vertices; i++)
    {
        if (adjacencies.find(i) != adjacencies.end())
        {
            for (auto &v : adjacencies[i])
            {
                col_idx[j]=v;
                weights[j]=(u_int8_t)generateRandomWeight();
                j++;
            }
        }
    }
}
int CSR::getNumberOfEdges(){
    return num_edges;
}
int CSR::getNumberofVertices()
{
    return num_vertices;
}

uint* CSR::getColIdx()
{
    return col_idx;
}
u_int8_t*CSR::getWeights()
{
    return weights;
}

uint*CSR::getRowPtr()
{
    return row_ptr;
}


CSR::~CSR()
{
    free(row_ptr);
    free(weights);
    free(col_idx);
}
// int main()
// {
//     std::string file = "/home/a1bderrahmane/SSSP_CUDA/datasets/simple-graph.txt";
//     return 0;
// }
