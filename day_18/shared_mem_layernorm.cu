#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define EPSILON 1e-6

__global__ void smem_layerNorm(float *X, float *P, int m, int n){

    __shared__ float smem[1024];
    
    int row = blockIdx.x; 
    int tidx = threadIdx.x;
    
    if(row < m) {
        float *row_in = X + row * n;
        float *row_out = P + row * n;

        float lmean = 0.0f;
        float lvar = 0.0f;

        for(int i = tidx; i < n; i += blockDim.x) {
            float a = row_in[i]; 
            lmean += a;
            lvar += (a * a);
        }

        __syncthreads();
        smem[tidx] = lmean; 
        __syncthreads();

        for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if(tidx < stride) smem[tidx] += smem[tidx + stride]; 
            __syncthreads();
        }

        float gmean = smem[0] / n;
        __syncthreads();

        smem[tidx] = lvar;
        __syncthreads();

        for(int stride = blockDim.x; stride > 0; stride /= 2) {
            if(tidx < stride) smem[tidx] += smem[tidx + stride];
            __syncthreads();
        }

        float gvar = (smem[0]/n) - (gmean * gmean);
        float stddev = rsqrtf(gvar + EPSILON); 
        __syncthreads();

        for(int i = tidx; i < n; i += blockDim.x)
            row_out[i] = (row_in[i] - gmean) * stddev;
    }
}
int main() {

    int M = 1024;int N = 1024;

    size_t size = M * N * sizeof(float);
    float *a, *b, *da, *db;

    a = (float*)malloc(size);
    b = (float*)malloc(size);

    for(int i = 0; i < M * N; i++) a[i] = i - 1;

    cudaMalloc((void**)&da, size);cudaMalloc((void**)&db, size);
    
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(M);

    smem_layerNorm<<<blocksPerGrid, threadsPerBlock>>>(da, db, M, N);

    cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);

    printf("Input matrix: \n");
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++) printf("%f ", a[i * N + j]);
        printf("\n");
    }

    printf("Output matrix: \n");
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++) printf("%f ", b[i * N + j]);
        printf("\n");
    }

    free(b);free(a);cudaFree(da);cudaFree(db);

}