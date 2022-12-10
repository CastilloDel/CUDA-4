/* 
   Daniel del Castillo de la Rosa
   
   High Performance Computing master - 2022/2023
   
   Heterogeneous programming

   Sum the rows of M NxN matrix
*/

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;

const unsigned int M = 2048;
const unsigned int N = 2048;
const unsigned int DEFAULT_BLOCK_SIZE = 128;  

vector<float> create_random_matrix(unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distribution(-1000.0, 1000.0);

    vector<float> matrix(size * size);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = distribution(gen);
    }
    return matrix;
}

vector<float> sum_rows_cpu(const vector<float>& matrix, unsigned int size) {
    vector<float> rows_sum(size, 0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            rows_sum[i] += matrix[i * size + j];
        }
    }
    return rows_sum;
}

__global__ void sum_rows_kernel(const float* const matrix, float* const rows_sum, int size) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= size) {
        return;
    }
    for(int column = 0; column < size; column++) {
        rows_sum[row] += matrix[row * size + column];
    }
}

vector<float> sum_rows_gpu(const vector<float>& matrix, int size, microseconds* const computation_duration) {
    float* gpu_matrix;
    unsigned int size_in_bytes = matrix.size() * sizeof(float);
    cudaMalloc(&gpu_matrix, size_in_bytes);
    cudaMemcpy(gpu_matrix, matrix.data(), size_in_bytes, cudaMemcpyHostToDevice); 

    float* gpu_rows_sum;
    unsigned int row_size_in_bytes = size * sizeof(float);
    cudaMalloc(&gpu_rows_sum, row_size_in_bytes);
    cudaMemset(gpu_rows_sum, 0, row_size_in_bytes); 

    dim3 dimBlock(DEFAULT_BLOCK_SIZE);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    
    auto start = steady_clock::now();

    sum_rows_kernel<<<dimGrid, dimBlock>>>(gpu_matrix, gpu_rows_sum, size);

    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    if (computation_duration != nullptr) {
        *computation_duration = duration_cast<microseconds>(end - start);
    }

    vector<float> rows_sum(size);
    cudaMemcpy(rows_sum.data(), gpu_rows_sum, row_size_in_bytes, cudaMemcpyDeviceToHost); 

    cudaFree(gpu_matrix);
    cudaFree(gpu_rows_sum);
    
    return rows_sum;
}

int main() {
    vector<float> matrix = create_random_matrix(N);
    
    vector<float> rows_sum_cpu = sum_rows_cpu(matrix, N);
    
    microseconds computation_duration;

    auto start = steady_clock::now();
    vector<float> rows_sum_gpu = sum_rows_gpu(matrix, N, &computation_duration);
    auto end = steady_clock::now();
    
    microseconds total_duration = duration_cast<microseconds>(end - start);

    bool equal = std::equal(rows_sum_cpu.begin(), rows_sum_cpu.end(), rows_sum_gpu.begin());
    std::cout << (equal ? "The result was the same for the CPU and GPU" : "There was an error");
    std::cout << "\n";

    std::cout << "Total time: " << total_duration.count() << "us\n";
    std::cout << "Computation time in the GPU: " << computation_duration.count() << "us\n";
}