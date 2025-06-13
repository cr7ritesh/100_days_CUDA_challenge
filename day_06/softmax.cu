#include <iostream>
#include <stdlib.h>
#include <math.h>

__global__ void expKernel(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = exp(a[i]);
}

__global__ void normalizeKernel(float* b, int n, float sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = b[i] / sum;
}

int main() {
    int N = 1024; 
    int size = N * sizeof(float);
    
    float* A = (float*)malloc(size); float* B = (float*)malloc(size);

    srand(23);
    for (int i = 0; i < N; ++i) A[i] = float(rand()) / RAND_MAX; 

    float *da, *db;
    cudaMalloc(&da, size); cudaMalloc(&db, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    expKernel<<<blockPerGrid, threadsPerBlock>>>(da, db, N);
    cudaMemcpy(B, db, size, cudaMemcpyDeviceToHost);

    float exp_sum = 0.0f;
    for (int i = 0; i < N; i++) exp_sum += B[i];

    normalizeKernel<<<blockPerGrid, threadsPerBlock>>>(db, N, exp_sum);
    cudaMemcpy(B, db, size, cudaMemcpyDeviceToHost);

    std::cout << "First few elements of input: ";
    for (int i = 0; i < 10; ++i) std::cout << A[i] << " ";
    std::cout << std::endl;

    std::cout << "First few elements of output: ";
    for (int i = 0; i < 10; ++i) std::cout << B[i] << " ";

    cudaFree(da);cudaFree(db);

    return 0;
}