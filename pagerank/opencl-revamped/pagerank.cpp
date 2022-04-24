#include<stdio.h>
#include <bits/stdc++.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>
#include <CL/cl.h>

using namespace std;

#define MAX_SOURCE_SIZE (0x10000000)

void get_adj_matrix(float* graph, int n, float d, FILE *inputFilePtr){

    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    int num, m, indexing;
    
    fscanf(inputFilePtr, "%d", &num);
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
            graph[destin* n + source] += d* 1.0  ;
        }
        else{
            graph[(destin - 1)* n + source - 1] += d* 1.0;
        }
    }
}


void manage_adj_matrix_serial(float* graph, int n){
    for(int j = 0; j < n; ++j){
        float sum = 0.0;

        for (int i = 0; i< n; ++i){
            sum += graph[i * n + j];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                graph[i * n + j] /= sum;
            }
            else{
                graph[i * n + j] = (1/(float)n);
            }
        }
    }

}

void initialize_rank_serial(float *r, int n) {
    for(int i = 0; i< n; ++i){
        r[i] = (1/(float)n);
    }
}

void store_rank_serial(float *r, float *r_last, int n) {
    for(int i = 0; i< n; ++i){
        r_last[i] = r[i];
    }
}

void matmul_serial(float *graph, float *r, float *r_last, int n) {
    for(int i = 0; i < n; ++i){
        float sum = 0.0;

        for (int j = 0; j< n; ++j){
            sum += r_last[j] * graph[i * n + j];
        }

        r[i] = sum;
    }
}

void rankdiff_serial(float *r, float *r_last, int n) {
    for(int i = 0; i< n; ++i){
        r_last[i] -= r[i];
    }
}

void top_nodes(float *r, int n, int count = 10){

    priority_queue<pair<float, int>> pq;

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
    inputFilePtr = fopen(inputfile, "r");
    int n = atoi(argv[2]); 

    float d = 0.85; //dmaping factor 

    float* graph = (float*)malloc(n * n * sizeof(float*));

    float* r = (float*) malloc(n * sizeof(float));
    float* r_last = (float*) malloc(n * sizeof(float));
    get_adj_matrix(graph, n, d, inputFilePtr);
    start = clock();

    FILE *fp;
    
    fp = fopen("pagerank_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_int err;

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (ret_num_devices <= 0) {
        printf("No OpenCL Device found. Exiting.\n");
        exit(0);
    }
    
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret); 
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_device_type dev_type;
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
    if (dev_type == CL_DEVICE_TYPE_GPU) {
        printf("usage of GPU confirmed \n");
    }

    cl_kernel manage_adj_matrix = clCreateKernel(program, "manage_adj_matrix" , &err);
    cl_kernel initialize_rank = clCreateKernel(program, "initialize_rank" , &err);
    cl_kernel store_rank = clCreateKernel(program, "store_rank" , &err); 
    cl_kernel matmul = clCreateKernel(program, "matmul" , &err);
    cl_kernel rank_diff = clCreateKernel(program, "rank_diff" , &err); 

    size_t global_size[1];
    size_t local_size[1];

    global_size[0] = n;
    local_size[0] = n;

    cl_mem gpu_graph, gpu_r, gpu_r_last;

    gpu_graph = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n * n, NULL, &err);
    err = clEnqueueWriteBuffer(command_queue, gpu_graph, CL_TRUE, 0, sizeof(cl_float) * n * n, graph, 0, NULL, NULL);

    err = clSetKernelArg(manage_adj_matrix, 0, sizeof(cl_mem), &gpu_graph);
    err |= clSetKernelArg(manage_adj_matrix, 1, sizeof(int), &n);
    err = clEnqueueNDRangeKernel(command_queue, manage_adj_matrix, 1, NULL, global_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    err = clEnqueueReadBuffer(command_queue, gpu_graph, CL_TRUE, 0, sizeof(cl_float) * n * n, graph, 0, NULL, NULL);
    clFinish(command_queue);
    
    gpu_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &err);
    err = clEnqueueWriteBuffer(command_queue, gpu_r, CL_TRUE, 0, sizeof(float) * n, r, 0, NULL, NULL);

    gpu_r_last = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    err = clEnqueueWriteBuffer(command_queue, gpu_r_last, CL_TRUE, 0, sizeof(float) * n, r_last, 0, NULL, NULL);

    err = clSetKernelArg(initialize_rank, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(initialize_rank, 1, sizeof(int), &n);

    err = clSetKernelArg(store_rank, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(store_rank, 1, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(store_rank, 2, sizeof(int), &n);  

    err = clSetKernelArg(matmul, 0, sizeof(cl_mem), &gpu_graph);
    err |= clSetKernelArg(matmul, 1, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(matmul, 2, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(matmul, 3, sizeof(int), &n);

    err = clSetKernelArg(rank_diff, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(rank_diff, 1, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(rank_diff, 2, sizeof(int), &n);

    err = clEnqueueNDRangeKernel(command_queue, initialize_rank, 1, NULL, global_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    int max_iter = 50;
    while(max_iter--){

        err = clEnqueueNDRangeKernel(command_queue, store_rank, 1, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(command_queue);
        
        err = clEnqueueNDRangeKernel(command_queue, matmul, 1, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(command_queue);

        err = clEnqueueNDRangeKernel(command_queue, rank_diff, 1, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(command_queue);
    }

    err = clEnqueueReadBuffer(command_queue, gpu_r, CL_TRUE, 0, sizeof(float) * n, r, 0, NULL, NULL);
    clFlush(command_queue);
    clReleaseMemObject(gpu_graph);
    clReleaseMemObject(gpu_r);
    clReleaseMemObject(gpu_r_last);

    clReleaseProgram(program);

    clReleaseKernel(manage_adj_matrix);
    clReleaseKernel(initialize_rank);
    clReleaseKernel(store_rank);
    clReleaseKernel(matmul);
    clReleaseKernel(rank_diff);


    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    end = clock();

    top_nodes(r, n);
    printf("Time taken :%f for OpenCL implementation with %d nodes.\n", float(end - start)/CLOCKS_PER_SEC, n);
    return 0;
}
