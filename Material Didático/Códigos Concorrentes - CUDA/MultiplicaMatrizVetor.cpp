#include <iostream>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1000

__global__ void matrixVectorMultiply(const float* matrix, const float* vector, float* result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < MATRIX_SIZE) {
        float sum = 0.0f;
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            sum += matrix[row * MATRIX_SIZE + col] * vector[col];
        }
        result[row] = sum;
    }
}

int main() {
    float* host_matrix = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* host_vector = new float[MATRIX_SIZE];
    float* host_result = new float[MATRIX_SIZE];
    
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            host_matrix[i * MATRIX_SIZE + j] = i + j;
        }
        host_vector[i] = i;
    }
    
    float* dev_matrix;
    float* dev_vector;
    float* dev_result;
    
    cudaMalloc((void**)&dev_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_vector, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, MATRIX_SIZE * sizeof(float));
    
    cudaMemcpy(dev_matrix, host_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vector, host_vector, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (MATRIX_SIZE + blockSize - 1) / blockSize;
    
    matrixVectorMultiply<<<gridSize, blockSize>>>(dev_matrix, dev_vector, dev_result);
    
    cudaMemcpy(host_result, dev_result, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        std::cout << host_result[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFree(dev_matrix);
    cudaFree(dev_vector);
    cudaFree(dev_result);
    
    delete[] host_matrix;
    delete[] host_vector;
    delete[] host_result;
    
    return 0;
}
