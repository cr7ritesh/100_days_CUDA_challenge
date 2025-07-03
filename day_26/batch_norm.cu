#include <stdio.h>

__global__ void batchNormForwardKernel(float *y, float *x, float *gamma, float *beta, 
                                       float *mean, float *variance, float epsilon, int n) 
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_hat = (x[idx] - mean[idx]) / sqrtf(variance[idx] + epsilon);
        y[idx] = gamma[idx] * x_hat + beta[idx];
    }
}

void batchNormForwardCPU(float *y_cpu, float *x, float *gamma, float *beta, float *mean,
                         float *variance, float epsilon, int n) 
{
    
    for (int i = 0; i < n; ++i) {
        float x_hat = (x[i] - mean[i]) / sqrtf(variance[i] + epsilon);
        y_cpu[i] = gamma[i] * x_hat + beta[i];
    }
}

int main() {
    
    int n = 1024 * 1024; 
    float epsilon = 1e-5f;
    size_t size = n * sizeof(float);
    
    float *x = (float *)malloc(size);
    float *gamma = (float *)malloc(size);
    float *beta = (float *)malloc(size);
    float *mean = (float *)malloc(size); 
    float *variance = (float *)malloc(size);
    float *y = (float *)malloc(size); 
    float *y_cpu = (float *)malloc(size); 

    for (int i = 0; i < n; ++i) 
    {
        x[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; 
        gamma[i] = (float)rand() / RAND_MAX * 2.0f;    
        beta[i] = (float)rand() / RAND_MAX - 0.5f;     
        mean[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; 
        variance[i] = (float)rand() / RAND_MAX;
    }

    float *d_x, *d_gamma, *d_beta, *d_mean, *d_variance, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_gamma, n * sizeof(float));
    cudaMalloc(&d_beta, n * sizeof(float));
    cudaMalloc(&d_mean, n * sizeof(float));
    cudaMalloc(&d_variance, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, variance, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    batchNormForwardKernel<<<gridSize, blockSize>>>(d_y, d_x, d_gamma, d_beta, d_mean, d_variance, epsilon, n);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    printf("Performing CPU verification...\n");
    batchNormForwardCPU(y_cpu, x, gamma, beta, mean, variance, epsilon, n);

    double maxError = 0.0;
    for (int i = 0; i < n; ++i) {
        double error = fabs(y[i] - y_cpu[i]);
        if (error > maxError) maxError = error;
    }

    printf("Max error between GPU and CPU results: %e\n", maxError);

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_mean); 
    cudaFree(d_variance);cudaFree(d_y);

    free(x); free(gamma); free(beta); free(mean); free(variance); free(y); free(y_cpu);

    float tolerance = 2e-4; 
    if (maxError > tolerance)
        printf("Verification FAILED!\n);
    else printf("Verification PASSED!\n");
}