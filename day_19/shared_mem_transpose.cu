#include <iostream>
#include <cstdlib>   

#define BLOCK_SIZE 4  

__global__ void matrixTransposeShared(float* input, float* output, int col, int row) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  

    int x_in = blockIdx.x * BLOCK_SIZE + threadIdx.x;  
    int y_in = blockIdx.y * BLOCK_SIZE + threadIdx.y;  

    int x_out = blockIdx.y * BLOCK_SIZE + threadIdx.x; 
    int y_out = blockIdx.x * BLOCK_SIZE + threadIdx.y; 

    if (x_in < col && y_in < row) 
        tile[threadIdx.y][threadIdx.x] = input[y_in * col + x_in];

    __syncthreads();  

    if (x_out < row && y_out < col)
        output[y_out * row + x_out] = tile[threadIdx.x][threadIdx.y];
}


int main() {
    int col = 8, row = 14;  
    int size = col * row * sizeof(float);

    float *a = new float[col * row];
    float *b = new float[col * row];

    for (int i = 0; i < col * row; i++)
        a[i] = (float)(std::rand() % 50); 

    float *da, *db;
    cudaMalloc((void**)&da, size);cudaMalloc((void**)&db, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((col + BLOCK_SIZE - 1) / BLOCK_SIZE, (row + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixTransposeShared<<<gridDim, blockDim>>>(da, db, col, row);

    cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);

    std::cout << "Original Matrix (First 5 Rows):\n";
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            std::cout << a[i * col + j] << " ";
        std::cout << std::endl;
    }

    std::cout << "\nTransposed Matrix (First 5 Rows):\n";
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++)
            std::cout << b[i * row + j] << "\t";
        std::cout << std::endl;
    }

    cudaFree(da);cudaFree(db);delete[] a;delete[] b;

    return 0;
}