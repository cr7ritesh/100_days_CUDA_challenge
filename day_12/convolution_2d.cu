#include <algorithm>
#include <iostream>
#include <cassert>

#define MASK_DIM 9
#define OFFSET (MASK_DIM / 2)

__constant__ int mask[MASK_DIM * MASK_DIM];

__global__ void conv2dKernel(int* input, int* res, int len_in) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int beg_r = row - OFFSET;
    int beg_c = col - OFFSET;
    
    int tmp = 0;
    for(int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++) {
            if((beg_r + i >= 0) && (beg_r + i < len_in) && (beg_c + j >= 0) && (beg_c + j < len_in)) 
                tmp += input[(beg_r + i) * len_in + (beg_c + j)] * mask[i * MASK_DIM + j];
        }
    }
    res[row * len_in + col] = tmp;
}

void verify_result(int *m, int *mask, int *result, int N) {
    int temp;

    int offset_r;
    int offset_c;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < MASK_DIM; k++) {
                offset_r = i - OFFSET + k;
                for (int l = 0; l < MASK_DIM; l++) {
                    offset_c = j - OFFSET + l;
                    if (offset_r >= 0 && offset_r < N) {
                        if (offset_c >= 0 && offset_c < N) 
                            temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
                    }
                }
            }
        assert(result[i * N + j] == temp);
    }
  }
}

int main() {
    int n = 1 << 10;
    int size = n * n * sizeof(int);
    int size_mask = MASK_DIM * MASK_DIM * sizeof(int);

    int *matrix = new int[n * n];
    int *ans = new int[n * n];
    int *h_mask = new int[MASK_DIM * MASK_DIM];

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) matrix[i * n + j] = rand() % 50;
    }

    for(int i = 0; i < MASK_DIM; i++) {
        for(int j = 0; j < MASK_DIM; j++) h_mask[i * MASK_DIM + j] = rand() % 10;
    }

    int *dmat;int *dans;
    cudaMalloc(&dmat, size);cudaMalloc(&dans, size);
    
    cudaMemcpy(dmat, matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, size_mask);

    dim3 block_dim(16, 16);
    dim3 grid_dim((n + 15) / 16, (n + 15) / 16);
    
    conv2dKernel<<<grid_dim, block_dim>>>(dmat, dans, n);

    cudaMemcpy(ans, dans, size, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(matrix, h_mask, ans, n);
    std::cout << "Result verified successfully!" << std::endl;
    
    cudaFree(dans);cudaFree(dmat);
    delete[] matrix;delete[] ans;delete[] h_mask;
    
    return 0;
}