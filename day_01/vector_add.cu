#include <iostream>
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) C[i] = B[i] + A[i];
}

int main() {
    int n = 10;

    float* A = new float[n];
    float* B = new float[n];
    float* C = new float[n];

    for(int i = 0; i < n; i++) {
        A[i] = i; B[i] = i * 2;
    }

    float *da, *db, *dc;

	cudaMalloc((void**) &da, sizeof(float) * n);
    cudaMalloc((void**) &db, sizeof(float) * n);
	cudaMalloc((void**) &dc, sizeof(float) * n);

	cudaMemcpy(da, A, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(db, B, sizeof(float) * n, cudaMemcpyHostToDevice);

	vecAddKernel<<<ceil(n/256.0), 256>>>(da, db, dc, n);

	cudaMemcpy(C, dc, sizeof(float) * n, cudaMemcpyDeviceToHost);

	cudaFree(da);cudaFree(db);cudaFree(dc);

    std::cout << "\nResults:\n";
    for(int i = 0; i < n; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    return 0;
}
