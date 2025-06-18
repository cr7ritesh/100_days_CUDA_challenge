#include <algorithm>
#include <iostream>

__global__ void conv1dKernel(int* input, int* mask, int* res, int len_in, int len_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int mid = len_mask / 2;
    int begin = tid - mid;
    int tmp = 0;
    for(int i = 0; i < len_mask; i++) {
        if(((begin + i) >= 0) && (begin + i < len_in)) tmp += input[begin + i] * mask[i];
    }
    res[tid] = tmp;
}

int main() {
    int n = 30;
    int size = n * sizeof(int);
    int m = 5;
    int size_mask = m * sizeof(int);
    
    std::vector<int> A(n);
    std::generate(begin(A), end(A), [](){ return rand() % 50; });
    
    std::vector<int> B(m);
    std::generate(begin(B), end(B), [](){ return rand() % 10; });
    
    std::vector<int> ans(n);
    
    int *da, *db, *dans;
    cudaMalloc(&da, size);
    cudaMalloc(&db, size_mask);
    cudaMalloc(&dans, size);
    
    cudaMemcpy(da, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), size_mask, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    conv1dKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dans, n, m);

    cudaMemcpy(ans.data(), dans, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("A(input):\n");
    for (int i = 0; i < n; i++) printf("%d ", A[i]);

    printf("\nB(mask):\n");
    for (int i = 0; i < m; i++) printf("%d ", B[i]);
    
    printf("\nans(output):\n");
    for (int i = 0; i < n; i++) printf("%d ", ans[i]);
    
    cudaFree(dans);cudaFree(db);cudaFree(da);
    
    return 0;
}