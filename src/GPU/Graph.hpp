#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/graphviz.hpp>
constexpr int INFINITY_WEIGHT = std::numeric_limits<int>::max();

struct Edge
{
    uint source;
    uint destination;
};

struct EdgeWeightProperty
{
    uint weight;
};
// Directed Graph
// No vertex property
// Edge Property
typedef boost::compressed_sparse_row_graph<boost::directedS, boost::no_property, EdgeWeightProperty> WeightedCSRGraph;
class Graph
{
public:
    Graph(const std::string &filename);
    // ~Graph();

    std::vector<std::pair<uint, uint>> edges;
    std::vector<EdgeWeightProperty> edge_properties;
    WeightedCSRGraph g;
    int num_vertices;
    int minWeight;
    int maxWeight;

private:
    void readfile(const std::string &filename);
    uint generateRandomWeight();
};
#endif
