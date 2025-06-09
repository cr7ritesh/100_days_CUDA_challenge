#include <iostream>

__global__ void MatrixMultiply(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

int main() {
    int N = 3;
    int *A, *B, *C;
    int size = N * N * sizeof(int);

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i + j;
            B[i * N + j] = i * j;
            C[i * N + j] = 0;
        }
    }

    int *da, *db,*dc;

    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    cudaMemcpy(da, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(1, 1);
    MatrixMultiply<<<dimGrid, dimBlock>>>(da, db, dc,N);

    cudaMemcpy(C, dc, size, cudaMemcpyDeviceToHost);
    
    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%d ", A[i * N + j]); 
        printf("\n"); 
    }
    printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%d ",B[i * N + j]); 
        printf("\n"); 
    }

    printf("C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%d ",C[i * N + j]); 
        printf("\n"); 
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(A);free(B);free(C);

}