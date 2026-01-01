// #include "GPUsolver.cuh"

// GPUsolver::GPUsolver(const std::string &filename)
// {
//     csr_graph = new CSR(filename);
//     col_idx = csr_graph->getColIdx();
//     row_ptr = csr_graph->getRowPtr();
//     weights = csr_graph->getWeights();
//     distances = csr_graph->getDistances();
// }

// void GPUsolver::allocate_device_memory()
// {
//     cudaMalloc(&col_idx_device, sizeof(col_idx));
//     cudaMalloc(&row_ptr_device, sizeof(row_ptr));
//     cudaMalloc(&weights_device, sizeof(weights_device));
//     cudaMalloc(&distances_device, sizeof(distances));

//     cudaMalloc(&nbVertices, sizeof(int));

//     cudaMemset(nbVertices, csr_graph->getNumberofVertices(), sizeof(int));
// }

// void GPUsolver::transfer_data_from_host_to_device()
// {
//     cudaMemcpy(col_idx_device, col_idx, sizeof(col_idx), cudaMemcpyHostToDevice);
//     cudaMemcpy(row_ptr_device, row_ptr, sizeof(row_ptr), cudaMemcpyHostToDevice);
//     cudaMemcpy(weights_device, weights, sizeof(weights), cudaMemcpyHostToDevice);
//     cudaMemcpy(distances_device, distances, sizeof(distances), cudaMemcpyHostToDevice);
// }

// void GPUsolver::transfer_data_from_device_to_host()
// {
//     cudaMemcpy(distances, distances_device, sizeof(distances), cudaMemcpyDeviceToHost);
// }
// void GPUsolver::workFrontSweepSolver(int source_node, thrust::device_vector<float> &d_distance)
// {
//     thrust::device_vector<uint> workFront(csr_graph->getNumberofVertices());
//     thrust::device_vector<uint> workFront_output(csr_graph->getNumberofVertices());

//     workFront[source_node] = 1;
//     thrust::fill(workFront_output.begin(), workFront_output.end(), 0);
//     while (thrust::reduce(workFront.begin(), workFront.end()))
//     {
//         uint *workFront_device = thrust::raw_pointer_cast(workFront.data());
//         uint *workFront_output_device = thrust::raw_pointer_cast(workFront.data());
//         workFrontSweep<<<GRID_SIZE, BLOCK_size>>>(workFront_device, workFront_output_device);
//         cudaDeviceSynchronize();
//         uint *temp = workFront_device;
//         workFront_device = workFront_output_device;
//         workFront_output_device = temp;
//         thrust::fill(workFront_output.begin(), workFront_output.end(), 0);
//     }
// }
// __global__ void GPUsolver::workFrontSweep(uint *workFront, uint *workFront_output)
// {

//     int global_thread_idx = utils::get_global_id();
//     int totalThreads = utils::get_total_threads();

//     for (int i = global_thread_idx; i < *nbVertices; i += totalThreads)
//     {
//         if (workFront[i] == 1)
//         {
//             for (int j = row_ptr_device[i]; j < row_ptr_device[i + 1]; j++)
//             {
//                 int destination = col_idx_device[j];
//                 int weight = weights_device[j];
//                 int new_dist = distances_device[i] + weight;
//                 int old_dist = atomicMin(&distances_device[destination], new_dist);
//                 if (new_dist < old_dist)
//                 {
//                     workFront_output[destination] = 1;
//                 }
//             }
//         }
//     }
// }

// GPUsolver::~GPUsolver()
// {
//     cudaFree(col_idx_device);
//     cudaFree(row_ptr_device);
//     cudaFree(weights_device);
//     cudaFree(distances_device);
// }
