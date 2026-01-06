#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include "CPUSolver.hpp"

namespace {
void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " -i <path/to/graph> [-o <output_file>] [-l <log_file>]\n"
              << "       " << prog << " --input_graph <path/to/graph> [--output <output_file>] [--log <log_file>]\n";
}
} // namespace

int main(int argc, char **argv) {
    std::string filename;
    std::string output_path;
    std::string log_path;
    const option long_opts[] = {
        {"input_graph", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"log", required_argument, nullptr, 'l'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:l:h", long_opts, nullptr)) != -1) {
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

    *log_stream << "Running CPU solver on: " << filename << "\n";

    CPUSolver solver(filename);
    solver.solve(0);
    *log_stream << "Computation complete. Writing results to "
                << (output_path.empty() ? "stdout" : output_path) << "\n";
    solver.printResults(*output_stream);
    return 0;
}
