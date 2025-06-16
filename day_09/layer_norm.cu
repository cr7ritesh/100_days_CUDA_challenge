#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


__global__ void LayerNormKernel(float *X, float *P, int m, int n){

    int row = threadIdx.x + (blockDim.x * blockIdx.x);

    if(row < m){

        float mean = 0.0f;
        float var = 0.0f;

        // get mean of that row
        for(int col = 0; col < n; col++) mean += X[row * n + col];    
        mean /= n;

        // get variance
        for(int col = 0; col < n; col++) var += (X[row * n + col] - mean) * (X[row * n + col] - mean);
        var /= n;

        // normalize each row
        float std_dev = sqrt(var + 1e-7);
        for(int col = 0; col < n; col++)
            P[row * n + col] = (X[row * n + col] - mean) / std_dev;
    }
}



int main() {
    int rows = 10, cols = 10;
    int size = rows * cols * sizeof(float);
    float *A, *B;

    A = (float*)malloc(size);
    B = (float*)malloc(size);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) A[i * cols + j] = rand() % 50;
    }

    float *da, *db;
    cudaMalloc(&da, size);cudaMalloc(&db, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_memory_size = cols * sizeof(float);
    LayerNormKernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(da, db, rows, cols);

    cudaMemcpy(B, db, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%.2f ", A[i * cols + j]);
        printf("\n");
    }

    printf("\nB:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%.2f ", B[i * cols + j]);
        printf("\n");
    }

    cudaFree(da);cudaFree(db);
    free(A);free(B);

    return 0;
}