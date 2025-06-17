#include <iostream>

__global__ void transposeMatrixKernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = y * width + x;
        int out_idx = x * height + y;
        output[out_idx] = input[in_idx];
    }
}

int main() {
    int width = 1024; int height = 1024;

    int size = width * height * sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);

    for (int i = 0; i < width * height; i++) A[i] = float(i % 50);

    float* da; float* db;
    cudaMalloc((void**)&da, size);cudaMalloc((void**)&db, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transposeMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, width, height);
    
    cudaMemcpy(B, db, size, cudaMemcpyDeviceToHost);
    
    bool flag = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (B[i * height + j] != A[j * width + i]) {
                flag = false;
                break;
            }
        }
    }

    std::cout << (flag ? "Transpose was successfull!" : "Failure!!");

    cudaFree(da);cudaFree(db);
    free(A);free(B);

    return 0;
}