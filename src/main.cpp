#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include "CPUSolver.hpp"
#include "GPUsolver.cuh"
#include "HybridSolver.cuh"

namespace {
void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " -i <path/to/graph> -v <CPU|GPU|Hybrid>\n"
              << "       " << prog << " --input_graph <path/to/graph> --solver_version <CPU|GPU|Hybrid>\n";
}
} // namespace

int main(int argc, char **argv) {
    std::string filename;
    std::string solverVersion;
    const option long_opts[] = {
        {"input_graph", required_argument, nullptr, 'i'},
        {"solver_version", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'}};

    // int opt;
    // while ((opt = getopt_long(argc, argv, "i:h:v", long_opts, nullptr)) != -1) {
    //     switch (opt) {
    //     case 'i':
    //         filename = optarg;
    //         break;
    //     case 'v':
    //         solverVersion = optarg;
    //         break;
    //     case 'h':
    //     default:
    //         print_usage(argv[0]);
    //         return (opt == 'h') ? 0 : 1;
    //     }
    // }

    // if (filename.empty() || solverVersion.empty()) {
    //     print_usage(argv[0]);
    //     return 1;
    // }
    filename = argv[2];
    HybridSolver solver(filename);
    solver.solve(0);
    solver.printDistances();
    
    return 0;
}
