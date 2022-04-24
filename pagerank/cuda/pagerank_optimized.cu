#include<stdio.h>
#include <bits/stdc++.h>
#include<cuda.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>

using namespace std;

__global__ void map_kernel(int *pages, float *page_ranks, float *maps, unsigned int *noutlinks, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    if(i < n){
      float outbound_rank = page_ranks[i]/(float)noutlinks[i];
      for(j=0; j<n; ++j){
          maps[i*n+j] = pages[i*n+j] == 0 ? 0.0f : pages[i*n+j]*outbound_rank;
      }
    }
}

 __global__ void init_pair_array(pair<float, int>* gpu_r_nodes, float * gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_nodes[id].first = gpu_r[id];
        gpu_r_nodes[id].second = id + 1;
    }
}

__global__ void reduce_kernel(float *page_ranks, float *maps, int n){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    float new_rank;
    float old_rank;

    if(j < n){
      old_rank = page_ranks[j];
      new_rank = 0.0f;
      for(i=0; i< n; ++i){
          new_rank += maps[i*n + j];
      }

      new_rank = ((1-0.85)/n)+(0./85*new_rank);
      page_ranks[j] = new_rank;
    }

}


void get_adj_matrix(int* pages, unsigned int *noutlinks, int n, FILE *inputFilePtr){

    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    int m, indexing;
    
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);
    
    for(int i = 0; i < n ; i++){
    
        for(int j = 0; j< n; ++j){
            pages[i* n + j] = 1;
            noutlinks[i] += 1;
        }
    }
}

void init_array(float *a, int n, float val){
    int i;
    for(i=0; i<n; ++i){
        a[i] = val;
    }
}

void top_nodes(float* r, int n, int nblocks, int BLOCKSIZE, int count = 10){

    pair<float, int> *r_nodes = (pair<float, int> *) malloc ( n * sizeof (pair<float, int>) );
    pair<float, int> *gpu_r_nodes;

    cudaMalloc(&gpu_r_nodes, n * sizeof (pair<float, int>));

    float* gpu_r;
    cudaMalloc(&gpu_r, sizeof(float)*n);
    cudaMemcpy(gpu_r, r, sizeof(float)*n, cudaMemcpyHostToDevice);

    init_pair_array<<<nblocks, BLOCKSIZE>>>(gpu_r_nodes, gpu_r, n);

    cudaMemcpy(r_nodes, gpu_r_nodes, n * sizeof (pair<float, int>), cudaMemcpyDeviceToHost);

    thrust::sort(thrust::host, r_nodes, r_nodes + n);

    int rank =1;
    while(rank <= count){
        printf("Rank %d Node is %d\n", rank, r_nodes[n - rank].second);
        rank++;
    }
}

int main(int argc, char** argv) {

    clock_t start, end;

    char *inputfile = argv[1];
    int n = atoi(argv[2]); 

    int *pages = (int*)malloc(n*n*sizeof(int));
    float *maps;
    float *page_ranks;
    unsigned int *noutlinks;

    FILE *inputFilePtr;
    inputFilePtr = fopen(inputfile, "r");

    page_ranks = (float*)malloc(sizeof(float)*n);
    maps = (float*)malloc(sizeof(float)*n*n);
    noutlinks = (unsigned int*)malloc(sizeof(unsigned int)*n);

    for (int i=0; i<n; i++) {
        noutlinks[i] = 0;
    }

    get_adj_matrix(pages, noutlinks, n, inputFilePtr);
    init_array(page_ranks, n, 1.0f / (float) n);

    start = clock();

    int *pages_device;
    float* page_ranks_device;
    float *maps_device;
    unsigned int *noutlinks_device;

    cudaMalloc(&pages_device, sizeof(int)*n*n);
    cudaMemcpy(pages_device, pages, sizeof(int)*n*n, cudaMemcpyHostToDevice);

    cudaMalloc(&page_ranks_device, sizeof(float)*n);
    cudaMemcpy(page_ranks_device, page_ranks, sizeof(float)*n, cudaMemcpyHostToDevice);

    cudaMalloc(&maps_device, sizeof(float)*n);
    cudaMemcpy(maps_device, maps, sizeof(float)*n, cudaMemcpyHostToDevice);

    cudaMalloc(&noutlinks_device, sizeof(unsigned int)*n);
    cudaMemcpy(noutlinks_device, noutlinks, sizeof(unsigned int)*n, cudaMemcpyHostToDevice);

    int nblocks = ceil(float(n) / 32);

    for(int t = 0; t < 1000; t++){
        map_kernel<<<nblocks, 32>>>(pages_device, page_ranks_device, maps_device, noutlinks_device, n);
        cudaDeviceSynchronize();
        reduce_kernel<<<nblocks, 32>>>(page_ranks_device, maps_device, n);
        cudaDeviceSynchronize();
    }

    end = clock();

    cudaMemcpy(page_ranks, page_ranks_device, sizeof(float)*n, cudaMemcpyDeviceToHost);
    top_nodes(page_ranks, n, nblocks, 32);
    printf("Time taken :%f for parallel implementation [CUDA] with %d nodes.\n", float(end - start)/CLOCKS_PER_SEC, n);
    return 0;
}