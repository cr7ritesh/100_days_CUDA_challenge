#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define AVERAGE_EDGES_PER_VERTEX 8

void generate_random_graph(int num_vertices, int* num_edges, int** edges, int** dest) {
    
    int max_edges = num_vertices * AVERAGE_EDGES_PER_VERTEX;
    *dest = (int*)malloc(max_edges * sizeof(int));
    *edges = (int*)malloc((num_vertices + 1) * sizeof(int));
    
    int current_edge = 0;
    (*edges)[0] = 0;
    
    for (int i = 0; i < num_vertices; i++) {
        int edges_for_vertex = rand() % (AVERAGE_EDGES_PER_VERTEX * 2);
        
        for (int j = 0; j < edges_for_vertex; j++) {
            int dest_vertex = rand() % num_vertices;
            if (dest_vertex != i) (*dest)[current_edge++] = dest_vertex;
        }
        (*edges)[i + 1] = current_edge;
    }
    
    *num_edges = current_edge;    
    *dest = (int*)realloc(*dest, current_edge * sizeof(int));
}

__global__ void bfs_kernel(int level, int num_vertices, int* edges, int* dest, int* labels, int* done) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_vertices && labels[tid] == level) {
        for (int edge = edges[tid]; edge < edges[tid + 1]; edge++) {
            int neighbor = dest[edge];
            if (atomicCAS(&labels[neighbor], -1, level + 1) == -1) atomicExch(done, 0);
        }
    }
}

void bfs_gpu(int source, int num_vertices, int num_edges, int* h_edges, int* h_dest, int* h_labels) {
    int *d_edges, *d_dest, *d_labels, *d_done;
    
    cudaMalloc((void**)&d_edges, (num_vertices + 1) * sizeof(int));
    cudaMalloc((void**)&d_dest, num_edges * sizeof(int));
    cudaMalloc((void**)&d_labels, num_vertices * sizeof(int));
    cudaMalloc((void**)&d_done, sizeof(int));
    
    cudaMemset(d_labels, -1, num_vertices * sizeof(int));
    
    cudaMemcpy(d_edges, h_edges, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    int initial_level = 0;
    cudaMemcpy(d_labels + source, &initial_level, sizeof(int), cudaMemcpyHostToDevice);
    
    int level = 0;
    int h_done;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (num_vertices + threadsPerBlock - 1) / threadsPerBlock;
    
    do {
        h_done = 1;  
        cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
        
        bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(level, num_vertices, d_edges, d_dest, d_labels, d_done);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        level++;
    } while (!h_done && level < num_vertices);
    
    cudaMemcpy(h_labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_edges);cudaFree(d_dest);
    cudaFree(d_labels);cudaFree(d_done);
}

int main() {
    int num_vertices = 10000;  
    int source = 0;           
    int num_edges;
    int *edges, *dest;
    
    generate_random_graph(num_vertices, &num_edges, &edges, &dest);
    printf("Graph generated with %d edges\n", num_edges);
    
    int *gpu_labels = (int*)malloc(num_vertices * sizeof(int));
    
    int max_level = -1;
    int unreachable = 0;
    for (int i = 0; i < num_vertices; i++) {
        if (gpu_labels[i] == -1) unreachable++;
    }
    
    printf("Unreachable vertices: %d (%.2f%%)\n", unreachable, (float)unreachable / num_vertices * 100);
    
    free(edges);free(dest);free(gpu_labels);
    
    return 0;
}