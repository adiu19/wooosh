#include<stdio.h>
#include <bits/stdc++.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>
#include <CL/cl.h>

using namespace std;

void top_nodes(float* r, int n){

    pair<float, int> *r_nodes = (pair<float, int> *) malloc ( n * sizeof (pair<float, int>) );

    for (int i = 0; i < n; i++){
        r_nodes[i].first = r[i];
        r_nodes[i].second = i + 1;
    }
    thrust::sort(thrust::host, r_nodes, r_nodes + n);

    int rank =1;
    while(rank <= 10){
        printf("Rank %d Node is %d\n", rank, r_nodes[n - rank].second);
        rank++;
    }
}


void manage_adj_matrix_serial(float* gpu_graph, int n){
    //int id = get_global_id(0);
    for (int id = 0; id < n; id++) {
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
}

void initialize_rank_serial(float* gpu_r, int n){
    for (int id = 0; id < n; id++) {
        if(id < n){
            gpu_r[id] = (1/(float)n);
        }
    }
}


void store_rank_serial(float* gpu_r,float* gpu_r_last, int n){
    for (int id = 0; id < n; id++) {
        if(id < n){
            gpu_r_last[id] = gpu_r[id];
        }
    }
}

void matmul_serial(float* gpu_graph, float* gpu_r, float* gpu_r_last, int n){
    for (int id = 0; id < n; id++) {
        if(id < n){
            float sum = 0.0;

            for (int j = 0; j< n; ++j){
                sum += gpu_r_last[j] * gpu_graph[id* n + j];
            }

            gpu_r[id] = sum;
        }
    }
}

void rank_diff_serial(float* gpu_r,float* gpu_r_last, int n){
    for (int id = 0; id < n; id++) {
        if(id < n){
            gpu_r_last[id] = fabs((float)gpu_r_last[id] - (float)gpu_r[id]);
        }
    }
}


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
            graph[i* n + j] = (1 - d)/float(n);
        }
    }

    while(m--){
        int source, destin;
        fscanf(inputFilePtr, "%d", &source);
        fscanf(inputFilePtr, "%d", &destin);
        if (indexing == 0){
            graph[destin* n + source] += d* 1.0;
        }
        else{
            graph[(destin - 1)* n + source - 1] += d* 1.0;
        }
    }
}

int main(int argc, char** argv){
    clock_t start, end;

    FILE *inputFilePtr;

    char * inputfile = argv[1];
    int n = atoi(argv[2]); 
    char * bsize = argv[3];
    int BLOCKSIZE = atoi(bsize);
    cl_int err;

    inputFilePtr = fopen(inputfile, "r");

    float* graph = (float*)malloc(n*n*sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));
    float* r_last = (float*) malloc(n * sizeof(float));

    float d = 0.85;
    get_adj_matrix(graph, n, d, inputFilePtr);   

    manage_adj_matrix_serial(graph, n);
    initialize_rank_serial(r, n);
    int max_iter = 1000;
    while(max_iter > 0){
        store_rank_serial(r, r_last, n);
        matmul_serial(graph, r, r_last, n);
        rank_diff_serial(r, r_last, n);
        max_iter -= 1;
    }

    end = clock();
    top_nodes(r, n);
    printf("Time taken :%f for parallel implementation with %d nodes.\n", float(end - start)/CLOCKS_PER_SEC, n);

    return 0;
}