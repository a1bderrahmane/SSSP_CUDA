#include "CSR.hpp"

namespace {
std::mt19937 csr_rng;
std::uniform_int_distribution<int> csr_weight_dist(minWeight, maxWeight);
bool csr_rng_seeded = false;

void ensure_csr_rng_seeded() {
    if (!csr_rng_seeded) {
        std::random_device dev;
        csr_rng.seed(dev());
        csr_rng_seeded = true;
    }
}
} // namespace

CSR::CSR(const std::string &filename)
{
    num_edges=0;
    makeAdjacencies(filename);
    allocateMemory();
    makeRowPtr();
    makeColIdx();
}
void CSR::initDistances()
{
    for(int i=0;i<getNumberofVertices();i++)
    {
        distances[i]=UINT32_MAX;
    }
}
void CSR::setRandomSeed(uint32_t seed) {
    csr_rng.seed(seed);
    csr_rng_seeded = true;
}

u_int8_t CSR::generateRandomWeight()
{
    ensure_csr_rng_seeded();
    return static_cast<u_int8_t>(csr_weight_dist(csr_rng));
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
    distances=(uint*)malloc(sizeof(uint)*num_vertices);

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

uint*CSR::getDistances()
{
    return distances;
}

int CSR::getAverageDegree()
{
    
    int degreeCumulativeSum=row_ptr[num_edges];
    return degreeCumulativeSum/num_edges;
}

// We generate the weights randomly following a uniform distribution
// So instead of getting the exact average we will just consider the 

// is this tre, on a sufficitly large graph it is, but for small graphs
// we can easly have significant bias here  ~ Borna
int CSR::getAverageEdgeWeight()
{
    return (maxWeight - minWeight) / 2;
}

CSR::~CSR()
{
    free(row_ptr);
    free(weights);
    free(col_idx);
    free(distances);
}
