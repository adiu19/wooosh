#include<stdio.h>
#include <bits/stdc++.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

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

    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("trial.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("platform id = %d \n", platform_id);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("device id  = %d \n", device_id);
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

    size_t local_size[1];// = { n < 32 ? n : 64};
    size_t global_size[1];// =  {n};//{ n % local_size[0] == 0 ? n : ((n/local_size[0])+1)*local_size[0]};

    global_size[0] = n * n;
    local_size[0] = n;

    printf("local = %d \n", local_size[0]);
    printf("global = %d \n", global_size[0]);

   // manage_adj_matrix_serial(graph, n);
    //initialize_rank_serial(r, n);
    cl_mem gpu_graph, gpu_r, gpu_r_last;

    gpu_graph = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n*n, NULL, &err);
    err = clEnqueueWriteBuffer(command_queue, gpu_graph, CL_TRUE, 0, sizeof(float) * n * n, graph, 0, NULL, NULL);
    printf("gpu graph transfer %d \n", err);

    err = clSetKernelArg(manage_adj_matrix, 0, sizeof(cl_mem), &gpu_graph);
    err |= clSetKernelArg(manage_adj_matrix, 1, sizeof(int), &n);

    err = clEnqueueNDRangeKernel(command_queue, manage_adj_matrix, 1, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(command_queue);
    printf("manage adj matrix done : %d \n", err);

    gpu_r = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    printf("gpu_r buffer created %d \n", err);
    err = clEnqueueWriteBuffer(command_queue, gpu_r, CL_TRUE, 0, sizeof(float) * n, r, 0, NULL, NULL);
    printf("gpu_r buffer enqueued %d\n", err);

    gpu_r_last = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    err = clEnqueueWriteBuffer(command_queue, gpu_r_last, CL_TRUE, 0, sizeof(float) * n, r, 0, NULL, NULL);
    printf("gpu_r_last buffer enqueued : %d \n", err);

    err = clSetKernelArg(initialize_rank, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(initialize_rank, 1, sizeof(int), &n);
    printf("init rank args : %d \n", err);

    err = clEnqueueNDRangeKernel(command_queue, initialize_rank, 1, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(command_queue);
    printf("ranks initialized : %d \n", err);


    err = clSetKernelArg(store_rank, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(store_rank, 1, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(store_rank, 2, sizeof(int), &n);  


    err = clSetKernelArg(matmul, 0, sizeof(cl_mem), &gpu_graph);
    err |= clSetKernelArg(matmul, 1, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(matmul, 2, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(matmul, 3, sizeof(int), &n);
   // clFinish(command_queue);

    err = clSetKernelArg(rank_diff, 0, sizeof(cl_mem), &gpu_r);
    err |= clSetKernelArg(rank_diff, 1, sizeof(cl_mem), &gpu_r_last);
    err |= clSetKernelArg(rank_diff, 2, sizeof(int), &n);
   // clFinish(command_queue);

    printf("args set : %d , powering up loop \n", err);
    int max_iter = 1000;
    while(max_iter > 0){
        err = clEnqueueNDRangeKernel(command_queue, store_rank, 1, NULL, global_size, local_size, 0, NULL, NULL);
        clFinish(command_queue);
       // store_rank_serial(r, r_last, n);
        //printf("%d \n", err);

       err = clEnqueueNDRangeKernel(command_queue, matmul, 1, NULL, global_size, local_size, 0, NULL, NULL);
       clFinish(command_queue);
        //printf("%d \n", err);
        //matmul_serial(graph, r, r_last, n);

        err = clEnqueueNDRangeKernel(command_queue, rank_diff, 1, NULL, global_size, local_size, 0, NULL, NULL);
        clFinish(command_queue);
        //printf("%d \n", err);
       // rank_diff_serial(r, r_last, n);

        err = clEnqueueReadBuffer(command_queue, gpu_r_last, CL_TRUE, 0, sizeof(float)*n, r_last, 0, NULL, NULL);
        clFinish(command_queue);

        //printf("%d \n", err);

        max_iter -= 1;
    }

    err = clEnqueueReadBuffer(command_queue, gpu_r, CL_TRUE, 0, sizeof(float)*n, r, 0, NULL, NULL);
    clFinish(command_queue);
    printf("enqueue of read command done : %d \n", err);

    end = clock();
    top_nodes(r, n);
    printf("Time taken :%f for parallel implementation with %d nodes.\n", float(end - start)/CLOCKS_PER_SEC, n);

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
    return 0;
}