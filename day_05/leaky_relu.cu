#include <iostream>

__global__ void LeakyRELU(float* a, float* b, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (a[i] >= 0) b[i] = a[i];
        else b[i] = alpha * a[i];
    }
}

int main() {
    int N = 10; 
    int size = N * sizeof(float);
    float alpha = 0.01f;

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);

    for (int i = 0; i < N; ++i) A[i] = float(i - 4); 

    float *da, *db;
    cudaMalloc(&da, size);cudaMalloc(&db, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = 1;

    LeakyRELU<<<gridSize, blockSize>>>(da, db, N, alpha);

    cudaMemcpy(B, db, size, cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < N; ++i) std::cout << A[i] << " ";
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < N; ++i) std::cout << B[i] << " ";

    cudaFree(da);cudaFree(db);

    return 0;
}