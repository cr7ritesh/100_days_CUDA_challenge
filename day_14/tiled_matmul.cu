#include <iostream>
#include <cuda.h>
#include <cassert>

#define TILE_WIDTH 32

__global__ void tiledMatrixMul(float* M, float* N, float* P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float tmp = 0.0f;

    for (int i = 0; i < Width / (float)TILE_WIDTH; i++){
        if (row < Width && i * TILE_WIDTH + tx < Width) 
            Mds[ty][tx] = M[row * Width + i * TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;

        if (col < Width && i * TILE_WIDTH + ty < Width) 
            Nds[ty][tx] = N[(i * TILE_WIDTH + ty) * Width + col]; 
        else Nds[ty][tx] = 0.0f;
        
        __syncthreads();

        for (int j =0; j < TILE_WIDTH; j++) tmp += Mds[ty][j] * Nds[j][tx];
        __syncthreads();
    }
    if (row < Width && col < Width) P[row * Width + col] = tmp;
}

__global__ void MatrixMul(float* M, float* N, float* P, int width){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row < width) && (col < width)){
        float tmp = 0.0f;
        for (int i = 0; i < width; i++) tmp += M[row * width + i] * N[i * width + col];
        P[row * width + col] = tmp;
    }
}


int main() {
    int n = 512;
    int size = n * n * sizeof(float);
    float *M = (float*)malloc(size);float *N = (float*)malloc(size);

    for(int i = 0; i < n ; i++) {
        for(int j = 0; j< n;j++) {
            M[i] = float(i - 1);
            N[i] = float(i + 1);
        }
    }
    

    float P1[n * n];float P2[n * n];
    float *dM, *dN, *dP1, *dP2;

    cudaMalloc((void**)&dM, size);cudaMalloc((void**)&dN, size);
    cudaMalloc((void**)&dP1, size);cudaMalloc((void**)&dP2, size);

    cudaMemcpy(dM, M, size, cudaMemcpyHostToDevice);cudaMemcpy(dN, N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);
    // std::cout<<"here!!"<<std::endl;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaDeviceSynchronize();
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    tiledMatrixMul<<<dimGrid, dimBlock>>>(dM, dN, dP1, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiled Matrix Multiplication Time: " << elapsedTime << " ms" << std::endl;

    cudaDeviceSynchronize();
    cudaEventDestroy(start);cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixMul<<<dimGrid, dimBlock>>>(dM, dN, dP2, n);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Naive Matrix Multiplication Time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);cudaEventDestroy(stop);

    cudaMemcpy(P1, dP1, size, cudaMemcpyDeviceToHost);cudaMemcpy(P2, dP2, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * n; i++) assert(fabs(P1[i] - P2[i]) < 1e-4);
    std::cout << "Multiplication Verified!!" << std::endl;

    cudaFree(dM);cudaFree(dN);cudaFree(dP1);cudaFree(dP2);
    
    return 0;
}