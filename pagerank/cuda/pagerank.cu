#include<stdio.h>
#include <bits/stdc++.h>
#include<cuda.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>

using namespace std;

int TPB = 32;
double kernel_execution_time = 0;
clock_t k_start, k_end;

void get_adj_matrix(float* graph, int n, float d, FILE *inputFilePtr){

    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    int m, indexing;
    
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);
    
    for(int i = 0; i< n ; i++){
    
        for(int j = 0; j< n; ++j){
            if (i * n + j > 2000000000) {
                printf("\taccessing index = %d \n", i* n + j);
            }
            graph[i* n + j] = (1 - d)/float(n);
        }
    }

    while(m--){
        int source, destin;
        fscanf(inputFilePtr, "%d", &source);
        fscanf(inputFilePtr, "%d", &destin);
        if (indexing == 0){
            graph[destin* n + source] += d* 1.0  ;
        }
        else{
            graph[(destin - 1)* n + source - 1] += d* 1.0;
        }
    }
}

__global__ void manage_adj_matrix(float* gpu_graph, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        float sum = 0.0;

        for (int i = 0; i< n; ++i){
            sum += gpu_graph[i* n + id];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                gpu_graph[i* n + id] /= sum;
            }
            else{
                gpu_graph[i* n + id] = (1/(float)n);
            }
        }
    }
}

__global__ void initialize_rank(float* gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r[id] = (1/(float)n);
    }
}

__global__ void store_rank(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = gpu_r[id];
    }
}

__global__ void matmul(float* gpu_graph, float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        float sum = 0.0;

        for (int j = 0; j< n; ++j){
            sum += gpu_r_last[j] * gpu_graph[id* n + j];
        }

        gpu_r[id] = sum;
    }
}

__global__ void rank_diff(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = abs(gpu_r_last[id] - gpu_r[id]);
    }
}

__global__ void init_pair_array(pair<float, int>* gpu_r_nodes, float * gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_nodes[id].first = gpu_r[id];
        gpu_r_nodes[id].second = id + 1;
    }
}


void power_method(float *graph, float *r, int n, int nblocks, int BLOCKSIZE, int max_iter = 50, float eps = 0.000001){
   
    float* r_last = (float*) malloc(n * sizeof(float));
    
    float* gpu_graph;
    cudaMalloc(&gpu_graph, sizeof(float)*n*n);
    cudaMemcpy(gpu_graph, graph, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    float* gpu_r;
    cudaMalloc(&gpu_r, sizeof(float)*n);

    float* gpu_r_last;
    cudaMalloc(&gpu_r_last, sizeof(float)*n);

    k_start = clock();
    initialize_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, n);
    cudaDeviceSynchronize();
    k_end = clock();

    kernel_execution_time += (double)(k_end - k_start)/CLOCKS_PER_SEC;

    while(max_iter--){
        k_start = clock();

        store_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        matmul<<<nblocks, BLOCKSIZE>>>(gpu_graph, gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();
        
        rank_diff<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        k_end = clock();

        kernel_execution_time += (double)(k_end - k_start)/CLOCKS_PER_SEC;

        cudaMemcpy(r_last, gpu_r_last, n* sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(r, gpu_r, n* sizeof(float), cudaMemcpyDeviceToHost);
    return;
}

void top_nodes(float *r, int n, int count = 10){

    priority_queue< pair<float, int> > pq;

    for(int i = 0; i< n; ++i){
        pq.push(make_pair(r[i], i+ 1));
    }
    int rank =1;
    while(rank <= count){
        printf("Rank %d Node is %d\n", rank, pq.top().second);
        rank++;
        pq.pop();
    }

}

int main(int argc, char** argv){

    clock_t start, end;

    FILE *inputFilePtr;
    char * inputfile = argv[1];

    int n; 
    inputFilePtr = fopen(inputfile, "r");

    fscanf(inputFilePtr, "%d", &n);

    int nblocks = ceil(float(n) / TPB);

    float* graph = (float*)malloc(n*n*sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));
    float d = 0.85;
    get_adj_matrix(graph, n, d, inputFilePtr);
    float* gpu_graph;

    start = clock();

    cudaMalloc(&gpu_graph, sizeof(float)*n*n);
    cudaMemcpy(gpu_graph, graph, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    k_start = clock();
    manage_adj_matrix<<<nblocks, TPB>>>(gpu_graph, n);
    cudaDeviceSynchronize();
    k_end = clock();
    kernel_execution_time += (double)(k_end - k_start)/CLOCKS_PER_SEC;

    cudaMemcpy(graph, gpu_graph, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

    power_method(graph, r, n, nblocks, TPB);
    cudaDeviceSynchronize();
    end = clock();

    top_nodes(r, n);
    printf("[CUDA] total time : %f, GPU time : %f with %d nodes.\n", double(end - start)/CLOCKS_PER_SEC, kernel_execution_time, n);
    return 0;
}