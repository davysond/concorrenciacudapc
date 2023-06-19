#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* a, const int* b, int* c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size = 1000;
    int* host_a = new int[size];
    int* host_b = new int[size];
    int* host_c = new int[size];
    
    for (int i = 0; i < size; ++i) {
        host_a[i] = i;
        host_b[i] = i;
    }
    
    int* dev_a;
    int* dev_b;
    int* dev_c;
    
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    
    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);
    
    cudaMemcpy(host_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i) {
        std::cout << host_c[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    
    return 0;
}
