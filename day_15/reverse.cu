#include <assert.h>
#include <stdio.h>
#include <cuda.h>

__global__ void reverseArray(int *a, int *b) {
    extern __shared__ int s[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    s[blockDim.x - 1 - threadIdx.x] = a[i];

    __syncthreads();

    int j = blockDim.x * (gridDim.x - 1 - blockIdx.x) + threadIdx.x;
    b[j] = s[threadIdx.x];
}

int main(int argc, char** argv) {
    int *a;
    int n = 256 * 1024;
    size_t size = n * sizeof(int);
    a = (int *)malloc(size);

    int *db, *da;
    cudaMalloc((void **) &da, size);cudaMalloc((void **) &db, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = n / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(int);

    for (int i = 0; i < n; i++) a[i] = i;

    cudaMemcpy( da, a, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(blocksPerGrid);dim3 dimBlock(threadsPerBlock);
    reverseArray <<< dimGrid, dimBlock, sharedMemSize >>> (da, db);
    
    cudaMemcpy(a, db, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) assert(a[i] == n - 1 - i);
    printf("Verified!!");
    cudaFree(da); cudaFree(db);free(a);
    return 0;
}