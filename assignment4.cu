/* 
   Daniel del Castillo de la Rosa
   
   High Performance Computing master - 2022/2023
   
   Heterogeneous programming

   Sum the rows of M NxN matrix
*/

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

const unsigned int M = 2048;
const unsigned int N = 2048;
const unsigned int DEFAULT_BLOCK_SIZE = 128;  

float* create_random_matrix(unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distribution(-1000.0, 1000.0);

    float* matrix = new float[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = distribution(gen);
        }
    }
    return matrix;
}

float* sum_rows_cpu(const float* const matrix, unsigned int size) {
    float* rows_sum = new float[size]();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            rows_sum[i] += matrix[i * SIZE + j];
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

float* sum_rows_gpu(const float* const matrix, unsigned int size, microseconds* const computation_duration) {
    float* gpu_matrix;
    unsigned int size_in_bytes = size * size * sizeof(float);
    cudaMalloc(&gpu_matrix, size_in_bytes);
    cudaMemcpy(gpu_matrix, matrix, size_in_bytes, cudaMemcpyHostToDevice); 

    float* gpu_rows_sum;
    cudaMalloc(&gpu_rows_sum, size_in_bytes);
    cudaMemset(gpu_rows_sum, 0, size_in_bytes); 

    dim3 dimBlock(DEFAULT_BLOCK_SIZE);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    
    auto start = steady_clock::now();

    sum_rows_kernel<<<dimGrid, dimBlock>>>(gpu_matrix, gpu_rows_sum, size);

    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    if (computation_duration != nullptr) {
        *computation_duration = duration_cast<microseconds>(end - start);
    }

    float* rows_sum = new float[size];
    cudaMemcpy(rows_sum, gpu_rows_sum, size * sizeof(float), cudaMemcpyDeviceToHost); 

    cudaFree(gpu_matrix);
    cudaFree(gpu_rows_sum);
    
    return rows_sum;
}

int main() {
    float* matrix = create_random_matrix(SIZE);
    
    float* rows_sum_cpu = sum_rows_cpu(matrix, SIZE);
    
    microseconds computation_duration;

    auto start = steady_clock::now();
    float* rows_sum_gpu = sum_rows_gpu(matrix, SIZE, &computation_duration);
    auto end = steady_clock::now();
    
    microseconds total_duration = duration_cast<microseconds>(end - start);

    bool equal = std::equal(rows_sum_cpu, rows_sum_cpu + SIZE, rows_sum_gpu);
    std::cout << (equal ? "The result was the same for the CPU and GPU" : "There was an error");
    std::cout << "\n";

    std::cout << "Total time: " << total_duration.count() << "us\n";
    std::cout << "Computation time in the GPU: " << computation_duration.count() << "us\n";

    delete[] matrix;
    delete[] rows_sum_cpu;
    delete[] rows_sum_gpu;
}