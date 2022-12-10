/* 
   Daniel del Castillo de la Rosa
   
   High Performance Computing master - 2022/2023
   
   Heterogeneous programming

   Sum the rows of M NxN matrixes_vector
*/

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;

const unsigned int M = 100;
const unsigned int N = 4;
const unsigned int DEFAULT_BLOCK_SIZE = 128;  

vector<float> create_random_matrix(unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distribution(-1000.0, 1000.0);

    vector<float> matrix(size * size);
    for (float& val : matrix) {
        val = distribution(gen);
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

vector<vector<float>> sum_matrixes_rows_cpu(const vector<vector<float>>& matrixes_vector, unsigned int size) {
    vector<vector<float>> rows_sum_vector(matrixes_vector.size());
    for (int i = 0; i < matrixes_vector.size(); i++) {
        rows_sum_vector[i] = sum_rows_cpu(matrixes_vector[i], size);
    }
    return rows_sum_vector;
}

__global__ void sum_rows_kernel(const float* const matrixes_vector, float* const rows_sum, int size) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= size) {
        return;
    }
    for (int column = 0; column < size; column++) {
        rows_sum[row] += matrixes_vector[row * size + column];
    }
}

vector<vector<float>> sum_matrixes_rows_gpu(const vector<vector<float>>& matrixes_vector, int size, microseconds* const computation_duration) {
    vector<vector<float>> rows_sum_vector(matrixes_vector.size());
    *computation_duration = microseconds(0);

    float* gpu_matrix;
    unsigned int size_in_bytes = size * size * sizeof(float);
    cudaMalloc(&gpu_matrix, size_in_bytes);

    float* gpu_rows_sum;
    unsigned int row_size_in_bytes = size * sizeof(float);
    cudaMalloc(&gpu_rows_sum, row_size_in_bytes);

    dim3 dimBlock(DEFAULT_BLOCK_SIZE);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    for (int i = 0; i < matrixes_vector.size(); i++) {
        cudaMemcpy(gpu_matrix, matrixes_vector[i].data(), size_in_bytes, cudaMemcpyHostToDevice); 
        cudaMemset(gpu_rows_sum, 0, row_size_in_bytes); 
        
        auto start = steady_clock::now();

        sum_rows_kernel<<<dimGrid, dimBlock>>>(gpu_matrix, gpu_rows_sum, size);

        cudaDeviceSynchronize();
        auto end = steady_clock::now();
        
        if (computation_duration != nullptr) {
            *computation_duration += duration_cast<microseconds>(end - start);
        }

        vector<float> rows_sum(size);
        cudaMemcpy(rows_sum.data(), gpu_rows_sum, row_size_in_bytes, cudaMemcpyDeviceToHost); 

        rows_sum_vector[i] = rows_sum; 
    }
    cudaFree(gpu_matrix);
    cudaFree(gpu_rows_sum);
    return rows_sum_vector;
}

int main() {
    vector<vector<float>> matrixes_vector(M);
    for (vector<float>& matrix : matrixes_vector) {
        matrix = create_random_matrix(N);
    }

    vector<vector<float>> rows_sum_cpu = sum_matrixes_rows_cpu(matrixes_vector, N);
    
    microseconds computation_duration;

    auto start = steady_clock::now();
    vector<vector<float>> rows_sum_gpu = sum_matrixes_rows_gpu(matrixes_vector, N, &computation_duration);
    auto end = steady_clock::now();
    
    microseconds total_duration = duration_cast<microseconds>(end - start);

    for (int i = 0; i < rows_sum_gpu.size(); i++) {
        if (!std::equal(rows_sum_cpu[i].begin(), rows_sum_cpu[i].end(), rows_sum_gpu[i].begin())) {
            std::cout << "There was an error: The results from the CPU and GPU doesn't match\n";
            break;
        }
    }

    std::cout << "Total time: " << total_duration.count() << "us\n";
    std::cout << "Computation time in the GPU: " << computation_duration.count() << "us\n";
}