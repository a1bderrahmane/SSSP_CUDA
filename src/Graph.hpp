#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <limits>
#include <iostream>

constexpr  int INFINITY_WEIGHT = std::numeric_limits<int>::max();

struct Edge {
    int destination;
    int weight;

    Edge(int dest = -1, int w = 0.0) : destination(dest), weight(w) {}
};

class Graph {
private:
    std::vector<std::vector<Edge>> adj_list;
    int num_nodes;
    long long num_edges;

public:

    Graph(int N) : num_nodes(N), num_edges(0) {
        adj_list.resize(N);
    }

    void addEdge(int src, int dest, int weight) {
        if (src >= 0 && src < num_nodes && dest >= 0 && dest < num_nodes) {
            adj_list[src].emplace_back(dest, weight);
            num_edges++;
        } else {
            std::cerr << "Warning: Invalid node ID(s) provided." << std::endl;
        }
    }
    int getNumNodes() const {
        return num_nodes;
    }
    long long getNumEdges() const {
        return num_edges;
    }

    const std::vector<Edge>& getEdges(int node) const {
        if (node >= 0 && node < num_nodes) {
            return adj_list[node];
        }
        static const std::vector<Edge> empty_edges;
        return empty_edges;
    }
};

#endif