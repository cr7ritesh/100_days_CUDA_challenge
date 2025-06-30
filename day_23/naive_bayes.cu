#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void calculate_class_stats(float* features, int* labels, 
                                      float* means, float* variances, int* class_counts,
                                      int num_samples, int num_features, int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_features * num_classes) {
        
        int feature_idx = idx % num_features;
        int class_idx = idx / num_features;
    
        float sum = 0.0f;
        float sum_squares = 0.0f;
        int count = 0;
    
        for (int i = 0; i < num_samples; i++) {
            if (labels[i] == class_idx) {
                float val = features[i * num_features + feature_idx];
                sum += val;
                sum_squares += val * val;
                count++;
            }
        }
    
        if (count > 0) {
            means[idx] = sum / count;
            variances[idx] = sum_squares / count - (sum / count) * (sum / count);
            class_counts[class_idx] = count;
        } 
        else {
            means[idx] = 0.0f;
            variances[idx] = 0.0f;
            class_counts[class_idx] = 0;
        }
    }
}

int main() {
    int num_samples = 6;
    int num_features = 2;
    int num_classes = 2;

    // Sample features (2D points)
    float features[] = {
        1.0f, 2.0f,  
        2.0f, 3.0f,  
        0.0f, 1.0f,  
        5.0f, 6.0f,  
        4.0f, 5.0f,  
        6.0f, 7.0f
    };

    int labels[] = {0, 0, 0, 1, 1, 1};

    float *dfeatures, *dmeans, *dvariances;
    int *dlabels, *dclass_counts;

    cudaMalloc(&dfeatures, num_samples * num_features * sizeof(float));
    cudaMalloc(&dlabels, num_samples * sizeof(int));
    cudaMalloc(&dmeans, num_classes * num_features * sizeof(float));
    cudaMalloc(&dvariances, num_classes * num_features * sizeof(float));
    cudaMalloc(&dclass_counts, num_classes * sizeof(int));

    cudaMemcpy(dfeatures, features, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dlabels, labels, num_samples * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((num_features * num_classes + block.x - 1) / block.x);
    
    calculate_class_stats<<<grid, block>>>(dfeatures, dlabels, dmeans, dvariances, dclass_counts,
                                           num_samples, num_features, num_classes);

    float means[num_classes * num_features]; 
    float variances[num_classes * num_features];
    int class_counts[num_classes];

    cudaMemcpy(means, dmeans, num_classes * num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(variances, dvariances, num_classes * num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(class_counts, dclass_counts, num_classes * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Class Counts:\n");
    for (int i = 0; i < num_classes; i++) 
        printf("Class %d: %d\n", i, class_counts[i]);

    printf("\nMeans:\n");
    for (int c = 0; c < num_classes; c++) {
        printf("Class %d: ", c);
        for (int f = 0; f < num_features; f++) 
            printf("%.2f ", means[c * num_features + f]);
        printf("\n");
    }

    printf("\nVariances:\n");
    for (int c = 0; c < num_classes; c++) {
        printf("Class %d: ", c);
        for (int f = 0; f < num_features; f++)
            printf("%.2f ", variances[c * num_features + f]);
        printf("\n");
    }

    cudaFree(dfeatures); cudaFree(dlabels);
    cudaFree(dmeans); cudaFree(dvariances);
    cudaFree(dclass_counts);

    return 0;
}