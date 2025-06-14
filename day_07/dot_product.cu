#include <iostream>
#include <stdlib.h>
#include <math.h>

#define get_min(a,b) (a<b?a:b) 
#define threadsPerBlock 256
#define N 1024

__global__ void dotKernel(float* a, float* b, float *c) {
    int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float dot[threadsPerBlock];
    int cache_ind = threadIdx.x;

    float tmp = 0.0f;
    while (th_id < N) {
        tmp += a[th_id] * b[th_id];
        th_id += blockDim.x * gridDim.x;
    }
    dot[cache_ind] = tmp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i) {
        if (cache_ind < i) dot[cache_ind] += dot[cache_ind + i];
        __syncthreads();
        i /= 2;
    }

    if (cache_ind == 0) c[blockIdx.x] = dot[0];
}

int main() {
    int size = N * sizeof(float);
    int blocksPerGrid = get_min(32, (N + threadsPerBlock - 1) / threadsPerBlock);
    
    float *a, *b, *res;
    float *da, *db, *dres;
    
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    res = (float*)malloc(blocksPerGrid * sizeof(float));
    
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dres, blocksPerGrid * sizeof(float));
    
    for (int i=0; i<N; i++) a[i] = i + 2,b[i] = i;

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    dotKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dres);
    cudaMemcpy(res, dres, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float out = 0.0f;
    for (int i=0; i<blocksPerGrid; i++) out += res[i];

    printf("%.10g", out);
    
}