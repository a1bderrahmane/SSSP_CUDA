#include "Graph.hpp"

void Graph::readfile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    uint u, v;
    uint w=generateRandomWeight();
    int max_index = -1;

    while (file >> u >> v)
    {
        edges.push_back({u, v});
        edge_properties.push_back({w});

        // Track the largest vertex index found
        if (u > max_index)
            max_index = u;
        if (v > max_index)
            max_index = v;
    }

    // The number of vertices is the maximum index + 1 (since indices are 0-based)
    num_vertices = max_index + 1;
}

Graph::Graph(const std::string &filename)
{
    readfile(filename);
    g =WeightedCSRGraph(
        boost::edges_are_unsorted_t(),        // Tag to use the unsorted constructor
        edges.begin(),                        // Start iterator for edges
        edges.end(),                          // End iterator for edges
        edge_properties.begin(),              // Start iterator for edge properties (weights)
        num_vertices                          // Total number of vertices
    );
}



uint Graph::generateRandomWeight()
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> gen(minWeight, maxWeight);
    return (uint)gen(rng);
}
// int main()
// {
//     std::string file="../datasets/simple-graph.txt";
//     Graph graph=Graph(file);
//     return 0;
// }

