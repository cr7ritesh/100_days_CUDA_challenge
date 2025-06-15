#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <numeric>

__global__ void reduceKernel(int *a, int *b) {
    extern __shared__ int sdata[];  

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = a[i];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) b[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 16;
    int a_size = N * sizeof(int);
    int b_size = ((N + 255) / 256) * sizeof(int);
    
    int *a, *b;
    int *da, *db;
    
    a = (int*)malloc(a_size);
    b = (int*)malloc(b_size);
    
    cudaMalloc((void**)&da, a_size);
    cudaMalloc((void**)&db, b_size);

    for (int i = 0; i < N; i++) a[i] = rand() % 50;

    cudaMemcpy(da, a, a_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    auto start = std::chrono::high_resolution_clock::now(); 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0; 
    
    cudaMemcpy(b, db, b_size, cudaMemcpyDeviceToHost);

    int finalResult = b[0];
    for (int i = 1; i < (N + 255) / 256; ++i) finalResult += b[i];

    int cpuResult = std::accumulate(a, a + N, 0);
    if (cpuResult == finalResult) {
        std::cout << "Verification successful: GPU result matches CPU result.\n";
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    } else {
        std::cout << "Verification failed: GPU result (" << finalResult << ") does not match CPU result (" << cpuResult << ").\n";
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    }
    
    std::cout << "Reduced result: " << finalResult << std::endl;
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;

    cudaFree(da);cudaFree(db);
    delete[] a;delete[] b;
    
}