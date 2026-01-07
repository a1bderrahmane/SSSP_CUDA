#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include "CPUSolver.hpp"
#include "GPUsolver.cuh"
#include "HybridSolver.cuh"

namespace {
void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " -i <path/to/graph> -S <CPU|GPU|Hybrid> [-o <output_file>] [-l <log_file>] [-n <source_node>] [--seed <value>]\n"
              << "       " << prog << " --input_graph <path/to/graph> --solver <CPU|GPU|Hybrid> [--output <output_file>] [--log <log_file>] [--node <source_node>] [--seed <value>]\n";
}

std::string to_upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

void write_distances(std::ostream &out, const std::vector<uint> &distances) {
    out << "node, distance_to\n";
    for (size_t i = 0; i < distances.size(); ++i) {
        out << i << ", " << distances[i] << "\n";
    }
}
} // namespace

int main(int argc, char **argv) {
    std::string filename;
    std::string solver;
    std::string output_path;
    std::string log_path;
    bool seed_provided = false;
    uint32_t seed_value = 0;
    uint source_node = 0;
    const option long_opts[] = {
        {"input_graph", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"node", required_argument, nullptr, 'n'},
        {"log", required_argument, nullptr, 'l'},
        {"seed", required_argument, nullptr, 's'},
        {"solver", required_argument, nullptr, 'S'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:l:n:s:S:h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'i':
            filename = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'l':
            log_path = optarg;
            break;
        case 'n': {
            char *end = nullptr;
            long value = std::strtol(optarg, &end, 10);
            if (optarg == end || *end != '\0' || value < 0) {
                std::cerr << "Invalid node id: " << optarg << "\n";
                return 1;
            }
            source_node = static_cast<uint>(value);
            break;
        }
        case 's': {
            char *end = nullptr;
            long value = std::strtol(optarg, &end, 10);
            if (optarg == end || *end != '\0' || value < 0) {
                std::cerr << "Invalid seed: " << optarg << "\n";
                return 1;
            }
            seed_provided = true;
            seed_value = static_cast<uint32_t>(value);
            break;
        }
        case 'S':
            solver = optarg;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    if (filename.empty() || solver.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::ofstream log_file;
    std::ostream *log_stream = &std::clog;
    if (!log_path.empty()) {
        log_file.open(log_path);
        if (!log_file) {
            std::cerr << "Failed to open log file: " << log_path << "\n";
            return 1;
        }
        log_stream = &log_file;
    }

    std::ofstream output_file;
    std::ostream *output_stream = &std::cout;
    if (!output_path.empty()) {
        output_file.open(output_path);
        if (!output_file) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        output_stream = &output_file;
    }

    if (seed_provided) {
        CSR::setRandomSeed(seed_value);
    }

    const std::string solver_upper = to_upper(solver);

    *log_stream << "Input graph: " << filename << "\n";
    *log_stream << "Selected solver: " << solver_upper << "\n";
    *log_stream << "Source node: " << source_node << "\n";
    if (seed_provided) {
        *log_stream << "Using RNG seed: " << seed_value << "\n";
    }

    if (solver_upper == "CPU") {
        CPUSolver solver(filename);
        solver.solve(source_node);
        *log_stream << "Computation complete. Writing results to "
                    << (output_path.empty() ? "stdout" : output_path) << "\n";
        solver.printResults(*output_stream);
    } else if (solver_upper == "GPU") {
        GPUsolver solver(filename);
        solver.solve(source_node);
        auto distances = solver.getDistancesHost();
        *log_stream << "Computation complete. Writing results to "
                    << (output_path.empty() ? "stdout" : output_path) << "\n";
        write_distances(*output_stream, distances);
    } else if (solver_upper == "HYBRID") {
        HybridSolver solver(filename);
        solver.solve(source_node);
        auto distances = solver.getDistances();
        *log_stream << "Computation complete. Writing results to "
                    << (output_path.empty() ? "stdout" : output_path) << "\n";
        write_distances(*output_stream, distances);
    } else {
        std::cerr << "Unknown solver version: " << solver << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
