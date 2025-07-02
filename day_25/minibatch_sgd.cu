#include <stdio.h>

__global__ void calculate_gradients_kernel(float* d_x_batch, float* d_y_true_batch, 
                                           float* d_w, float* d_b, float* d_gradients_w, 
                                           float* d_gradients_b, int batch_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size) {
        float x = d_x_batch[tid];
        float y_true = d_y_true_batch[tid];
        
        float w = *d_w;
        float b = *d_b;
        
        float y_pred = w * x + b;
        
        float error = y_pred - y_true;
        
        d_gradients_w[tid] = 2.0f * error * x;
        d_gradients_b[tid] = 2.0f * error;
    }
}

__global__ void reduce_sum_kernel(float* d_input, float* d_output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) d_output[blockIdx.x] = sdata[0];
}

void generate_data(float* x, float* y, int n, float true_w, float true_b, float noise_scale) {
    
    for (int i = 0; i < n; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        
        float noise = ((float)rand() / RAND_MAX) * 2.0f * noise_scale - noise_scale;
        y[i] = true_w * x[i] + true_b + noise;
    }
}

float calculate_mse(float* x, float* y_true, float w, float b, int n) {
    float mse = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float y_pred = w * x[i] + b;
        float error = y_pred - y_true[i];
        mse += error * error;
    }
    
    return mse / n;
}

void shuffle_data(float* x, float* y, int n) {
    
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        float temp_x = x[i];
        x[i] = x[j];
        x[j] = temp_x;
        
        float temp_y = y[i];
        y[i] = y[j];
        y[j] = temp_y;
    }
}

int main() {
    int data_size = 10000;
    int batch_size = 128;
    int num_epochs = 100;
    float learning_rate = 0.01f;
    
    float true_w = 2.5f;
    float true_b = 1.2f;
    float noise_scale = 0.5f;
    
    printf("Mini-Batch SGD for Linear Regression\n");
    printf("-------------------------------------\n");
    printf("Data size: %d\n", data_size);
    printf("Batch size: %d\n", batch_size);
    printf("Number of epochs: %d\n", num_epochs);
    printf("Learning rate: %.4f\n", learning_rate);
    printf("True parameters: w = %.2f, b = %.2f\n", true_w, true_b);
    printf("Noise scale: %.2f\n", noise_scale);
    printf("-------------------------------------\n\n");
    
    
    float* x_full = (float*)malloc(data_size * sizeof(float));
    float* y_true_full = (float*)malloc(data_size * sizeof(float));
    
    generate_data(x_full, y_true_full, data_size, true_w, true_b, noise_scale);
    
    float* x_batch = (float*)malloc(batch_size * sizeof(float));
    float* y_true_batch = (float*)malloc(batch_size * sizeof(float));
    
    
    float w = 0.0f;  
    float b = 0.0f;  
    
    float sum_gradient_w = 0.0f;
    float sum_gradient_b = 0.0f;
    
    float *dx_batch, *dy_true_batch;
    cudaMalloc((void**)&dx_batch, batch_size * sizeof(float));
    cudaMalloc((void**)&dy_true_batch, batch_size * sizeof(float));
    
    float *dw, *db;
    cudaMalloc((void**)&dw, sizeof(float));
    cudaMalloc((void**)&db, sizeof(float));
    
    float *dgradients_w, *dgradients_b;
    cudaMalloc((void**)&dgradients_w, batch_size * sizeof(float));
    cudaMalloc((void**)&dgradients_b, batch_size * sizeof(float));
    
    float *dsum_gradient_w, *dsum_gradient_b;

    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    cudaMalloc((void**)&dsum_gradient_w, num_blocks * sizeof(float));
    cudaMalloc((void**)&dsum_gradient_b, num_blocks * sizeof(float));
    
    float *dfinal_sum_w = NULL, *dfinal_sum_b = NULL;
    if (num_blocks > 1) {
        cudaMalloc((void**)&dfinal_sum_w, sizeof(float));
        cudaMalloc((void**)&dfinal_sum_b, sizeof(float));
    }
    
    printf("Starting training...\n");
    
    float initial_loss = calculate_mse(x_full, y_true_full, w, b, data_size);
    printf("Initial loss: %.6f\n", initial_loss);

    dim3 blockDim(block_size);
    dim3 gridDim(num_blocks);
    
    dim3 finalGridDim(1);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        
        shuffle_data(x_full, y_true_full, data_size);
        
        for (int batch_start = 0; batch_start < data_size; batch_start += batch_size) {
            
            int current_batch_size = batch_size;
            if (batch_start + batch_size > data_size) 
                current_batch_size = data_size - batch_start;
            
            for (int i = 0; i < current_batch_size; i++) {
                x_batch[i] = x_full[batch_start + i];
                y_true_batch[i] = y_true_full[batch_start + i];
            }
            
            cudaMemcpy(dx_batch, x_batch, current_batch_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dy_true_batch, y_true_batch, current_batch_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dw, &w, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);
            
            calculate_gradients_kernel<<<gridDim, blockDim>>>(dx_batch, dy_true_batch, dw, db, 
                                                              dgradients_w, dgradients_b, current_batch_size);
            
            reduce_sum_kernel<<<num_blocks, blockDim>>>(dgradients_w, dsum_gradient_w, current_batch_size);
            reduce_sum_kernel<<<num_blocks, blockDim>>>(dgradients_b, dsum_gradient_b, current_batch_size);
            
            if (num_blocks > 1) {
                reduce_sum_kernel<<<finalGridDim, blockDim>>>(dsum_gradient_w, dfinal_sum_w, num_blocks);
                reduce_sum_kernel<<<finalGridDim, blockDim>>>(dsum_gradient_b, dfinal_sum_b, num_blocks);
                
                cudaMemcpy(&sum_gradient_w, dfinal_sum_w, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&sum_gradient_b, dfinal_sum_b, sizeof(float), cudaMemcpyDeviceToHost);
            } 
            else {
                cudaMemcpy(&sum_gradient_w, dsum_gradient_w, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&sum_gradient_b, dsum_gradient_b, sizeof(float), cudaMemcpyDeviceToHost);
            }
            
            float avg_gradient_w = sum_gradient_w / current_batch_size;
            float avg_gradient_b = sum_gradient_b / current_batch_size;
            
            w -= learning_rate * avg_gradient_w;
            b -= learning_rate * avg_gradient_b;
        }
        
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            float current_loss = calculate_mse(x_full, y_true_full, w, b, data_size);
            printf("Epoch %d/%d - Loss: %.6f, w: %.4f, b: %.4f\n", 
                   epoch + 1, num_epochs, current_loss, w, b);
        }
    }
    
    float final_loss = calculate_mse(x_full, y_true_full, w, b, data_size);
    
    printf("\nTraining completed!\n");
    printf("-------------------------------------\n");
    printf("Initial parameters: w = %.4f, b = %.4f\n", 0.0f, 0.0f);
    printf("Learned parameters: w = %.4f, b = %.4f\n", w, b);
    printf("True parameters:    w = %.4f, b = %.4f\n", true_w, true_b);
    printf("Final loss: %.6f\n", final_loss);
    printf("-------------------------------------\n");
    
    float w_error = fabs((w - true_w) / true_w) * 100.0f;
    float b_error = fabs((b - true_b) / true_b) * 100.0f;
    printf("Relative error: w = %.2f%%, b = %.2f%%\n", w_error, b_error);
    
    cudaFree(dx_batch); cudaFree(dy_true_batch); cudaFree(dw); cudaFree(db);
    cudaFree(dgradients_w); cudaFree(dgradients_b); cudaFree(dsum_gradient_w); 
    cudaFree(dsum_gradient_b);
    
    if (num_blocks > 1) {
        cudaFree(dfinal_sum_w);
        cudaFree(dfinal_sum_b);
    }
    
    free(x_full); free(y_true_full);
    free(x_batch); free(y_true_batch);
    
    printf("\nMini-Batch SGD finished\n");
}