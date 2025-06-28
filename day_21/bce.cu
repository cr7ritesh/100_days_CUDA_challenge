#include <stdio.h>
#include <cuda_runtime.h>

__global__ void binaryCrossEntropyKernel(float *ytrue, float *ypred, float *loss, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float y = ytrue[idx];
        float y_hat = ypred[idx];

        // Avoid log(0)
        y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7), 1e-7);

        float sample_loss = -(y * logf(y_hat) + (1 - y) * logf(1 - y_hat));
        atomicAdd(loss, sample_loss);
    }
}

float binaryCrossEntropyCUDA(float *y_true, float *y_pred, int N) {
    float *dtrue, *dpred, *dloss;
    float loss = 0.0f;
    size_t size = N * sizeof(float);

    cudaMalloc(&dtrue, size);
    cudaMalloc(&dpred, size);
    cudaMalloc(&dloss, sizeof(float));

    cudaMemcpy(dtrue, y_true, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dpred, y_pred, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dloss, &loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    binaryCrossEntropyKernel<<<blocksPerGrid, threadsPerBlock>>>(dtrue, dpred, dloss, N);
    cudaMemcpy(&loss, dloss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dtrue);cudaFree(dpred);cudaFree(dloss);
    return loss / N;
}

int main() {
    int N = 10;
    float y_true[N] = {1, 0, 1, 1, 0, 1, 0, 1, 0, 1};
    float y_pred[N] = {0.9, 0.1, 0.8, 0.7, 0.2, 0.85, 0.05, 0.95, 0.15, 0.98};

    float cpu_loss = 0.0f;
    for (int i = 0; i < N; i++) {
        float y_hat = y_pred[i];
        float y = y_true[i];
        y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7), 1e-7);
        cpu_loss += (-(y * logf(y_hat) + (1 - y) * logf(1 - y_hat)));
    }

    float loss = binaryCrossEntropyCUDA(y_true, y_pred, N);
    printf("GPU Binary Cross Entropy Loss: %f\n", loss);
    printf("CPU Binary Cross Entropy Loss: %f\n", cpu_loss / N);

    return 0;
}