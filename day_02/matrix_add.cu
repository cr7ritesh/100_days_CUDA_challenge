#include <iostream>

__global__ void MatrixAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < N) && (j < N)) C[i * N + j] = A[i * N + j] + B[i * N + j];
}

int main() {
    int N = 10;
    float *A, *B, *C;

    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i + j;
            B[i * N + j] = i * j;
            C[i * N + j] = 0.0f;
        }
    }

    float *da, *db,*dc;

    cudaMalloc((void **)&da,N * N * sizeof(float));
    cudaMalloc((void **)&db,N * N * sizeof(float));
    cudaMalloc((void **)&dc,N * N * sizeof(float));

    cudaMemcpy(da,A,N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(db,B,N * N * sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(1, 1);
    MatrixAdd<<<dimGrid, dimBlock>>>(da, db, dc,N);

    cudaMemcpy(C,dc,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    
    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%.2f ", A[i * N + j]); 
        printf("\n"); 
    }
    printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%.2f ", B[i * N + j]); 
        printf("\n"); 
    }

    printf("C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%.2f ",C[i * N + j]); 
        printf("\n"); 
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

}