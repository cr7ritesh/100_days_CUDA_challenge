#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmaxCrossEntropyKernel(float *y_true, float *X, float *W, float *loss, int N, int D) {
    extern __shared__ float sharedMem[];
    float *logits = sharedMem;

    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= N) return;

    int true_label = (int)y_true[sample_idx];

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        logits[d] = 0.0f;
        for (int j = 0; j < D; j++) logits[d] += X[sample_idx * D + j] * W[j * D + d];
    }
    __syncthreads();

    float max_logit = -1e20f;
    for (int d = 0; d < D; d++) max_logit = fmaxf(max_logit, logits[d]);

    float sum_exp = 0.0f;
    for (int d = 0; d < D; d++) {
        logits[d] = expf(fmaxf(logits[d] - max_logit, -50.0f));
        sum_exp += logits[d];
    }

    for (int d = 0; d < D; d++) logits[d] /= sum_exp;

    float sample_loss = -logf(fmaxf(logits[true_label], 1e-7));

    __shared__ float block_loss;
    if (threadIdx.x == 0) block_loss = 0.0f;
    __syncthreads();

    atomicAdd(&block_loss, sample_loss);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(loss, block_loss);
}


int main() {
    int N = 1000;
    int D = 10;

    float *ytrue = (float*)malloc(N * sizeof(float));
    float *X = (float*)malloc(N * D * sizeof(float));
    float *W = (float*)malloc(D * D * sizeof(float));

    for (int i = 0; i < N; i++) {
        ytrue[i] = rand() % D;
        for (int d = 0; d < D; d++) X[i * D + d] = (float)(rand()) / RAND_MAX;
    }
    for (int d = 0; d < D * D; d++) W[d] = (float)(rand()) / RAND_MAX;

    float *dytrue, *dX, *dW, *dloss;
    float loss = 0.0f;

    cudaMalloc(&dytrue, N * sizeof(float)); cudaMalloc(&dX, N * D * sizeof(float));
    cudaMalloc(&dW, D * D * sizeof(float)); cudaMalloc(&dloss, sizeof(float));

    cudaMemcpy(dytrue, ytrue, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W, D * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dloss, &loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = D * sizeof(float);

    softmaxCrossEntropyKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(dytrue, dX, dW, dloss, N, D);

    cudaMemcpy(&loss, dloss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dytrue);cudaFree(dX);
    cudaFree(dW);cudaFree(dloss);

    printf("Optimized Softmax Cross-Entropy Loss: %f \n", loss);

    free(ytrue);free(X);free(W);

    return 0;
}