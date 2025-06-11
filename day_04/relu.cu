#include <iostream>

__global__ void RELU(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = fmaxf(0.0f, a[i]);
}

int main() {
    int N = 10; 
    int size = N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);

    for (int i = 0; i < N; ++i) A[i] = float(i - 4); 

    float *da, *db;
    cudaMalloc(&da, size);cudaMalloc(&db, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = 1;

    RELU<<<gridSize, blockSize>>>(da, db, N);

    cudaMemcpy(B, db, size, cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < N; ++i) std::cout << A[i] << " ";
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < N; ++i) std::cout << B[i] << " ";

    cudaFree(da);cudaFree(db);

    return 0;
}