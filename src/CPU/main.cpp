#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include "CPUSolver.hpp"

namespace {
void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " -i <path/to/graph>\n"
              << "       " << prog << " --input_graph <path/to/graph>\n";
}
} // namespace

int main(int argc, char **argv) {
    std::string filename;
    const option long_opts[] = {
        {"input_graph", required_argument, nullptr, 'i'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'i':
            filename = optarg;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    if (filename.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    CPUSolver solver(filename);
    solver.solve(0);
    solver.printResults();
    return 0;
}
