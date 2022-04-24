#include <stdio.h>
#include <stdlib.h>
 #include <time.h>
 #include <math.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

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

void init_array(float *a, int n, float val){
    int i;
    for(i=0; i<n; ++i){
        a[i] = val;
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

    cl_int err;
    int t;
    page_ranks = (float*)malloc(sizeof(float)*n);
    maps = (float*)malloc(sizeof(float)*n*n);
    noutlinks = (unsigned int*)malloc(sizeof(unsigned int)*n);

    for (int i=0; i<n; i++) {
        noutlinks[i] = 0;
    }

    get_adj_matrix(pages, noutlinks, n, inputFilePtr);
    init_array(page_ranks, n, 1.0f / (float) n);

    int nb_links = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            nb_links += pages[i*n+j];
        }
    }
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("pagerank_kernel.cl", "r");
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
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (ret_num_devices <= 0) {
        printf("No OpenCL Device found. Exiting.\n");
        exit(0);
    }
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel map_kernel = clCreateKernel(program, "map_page_rank" , &err); // Create the compute kernel in the program we wish to run
    cl_kernel reduce_kernel = clCreateKernel(program, "reduce_page_rank" , &err); // Create the compute kernel in the program we wish to run

    float *diffs, *nzeros;
    diffs  =(float *) malloc(sizeof(float)*n);
    nzeros = (float *)malloc(sizeof(float)*n);
    for(int i = 0; i < n; i++){
      diffs[i] = 0.0f;
      nzeros[i] = 0.0f;
    }

    cl_mem page_ranks_d, maps_d, noutlinks_d, pages_d, dif_d;

    pages_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*pages)*n*n, NULL, &err);
    page_ranks_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(*page_ranks)*n, NULL, &err);
    maps_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(*maps)*n*n, NULL, &err);
    noutlinks_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*noutlinks)*n, NULL, &err);
    dif_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(*diffs)*n, NULL, &err);

    err = clEnqueueWriteBuffer(command_queue, pages_d, CL_TRUE, 0, sizeof(*pages)*n*n, pages, 0, NULL, NULL);
    clFinish(command_queue);
    err = clEnqueueWriteBuffer(command_queue, maps_d, CL_TRUE, 0, sizeof(*maps)*n*n, maps, 0, NULL, NULL);
    clFinish(command_queue);
    err = clEnqueueWriteBuffer(command_queue, page_ranks_d, CL_TRUE, 0, sizeof(*page_ranks)*n, page_ranks, 0, NULL, NULL);
    clFinish(command_queue);
    err = clEnqueueWriteBuffer(command_queue, noutlinks_d, CL_TRUE, 0, sizeof(*noutlinks)*n, noutlinks, 0, NULL, NULL);
    clFinish(command_queue);
    err = clEnqueueWriteBuffer(command_queue, dif_d, CL_TRUE, 0, sizeof(*diffs)*n, diffs, 0, NULL, NULL);
    clFinish(command_queue);
 
    int nblocks = ceil(float(n) / 32);
    size_t local_size[1] = { n < 32 ? n : 32};
    size_t global_size[1] = {nblocks};//{ n % local_size[0] == 0 ? n : ((n/local_size[0])+1)*local_size[0]};

        err = clSetKernelArg(map_kernel, 0, sizeof(cl_mem), &pages_d);
        err |= clSetKernelArg(map_kernel, 1, sizeof(cl_mem), &page_ranks_d);
        err |= clSetKernelArg(map_kernel, 2, sizeof(cl_mem), &maps_d);
        err |= clSetKernelArg(map_kernel, 3, sizeof(cl_mem), &noutlinks_d);
        err |= clSetKernelArg(map_kernel, 4, sizeof(int), &n);
        clFinish(command_queue);

        err = clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &page_ranks_d);
        err |= clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &maps_d);
        err |= clSetKernelArg(reduce_kernel, 2, sizeof(int), &n);
        err |= clSetKernelArg(reduce_kernel, 3, sizeof(cl_mem), &dif_d);
        clFinish(command_queue);

    start = clock();

    for(t=1; t <= 1000; ++t){
        // // MAP PAGE RANKS
        err = clEnqueueNDRangeKernel(command_queue, map_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
        clFinish(command_queue);

        // REDUCE PAGE RANKS
        err = clEnqueueNDRangeKernel(command_queue, reduce_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
        clFinish(command_queue);

        err = clEnqueueReadBuffer(command_queue, dif_d, CL_TRUE, 0, sizeof(float)*n, diffs, 0, NULL, NULL);
        clFinish(command_queue);
        err = clEnqueueWriteBuffer(command_queue, dif_d, CL_TRUE, 0, sizeof(*nzeros)*n, nzeros, 0, NULL, NULL);
        clFinish(command_queue);
    }

    err = clEnqueueReadBuffer(command_queue, maps_d, CL_FALSE, 0,  sizeof(*maps)*n*n, maps, 0, NULL, NULL);
    err = clEnqueueReadBuffer(command_queue, page_ranks_d, CL_FALSE, 0, sizeof(*page_ranks)*n, page_ranks, 0, NULL, NULL);
    clFinish(command_queue);

    end = clock();

    free(pages);
    free(maps);
    free(page_ranks);
    free(noutlinks);

    clReleaseMemObject(pages_d);
    clReleaseMemObject(maps_d);
    clReleaseMemObject(page_ranks_d);
    clReleaseMemObject(noutlinks_d);
    clReleaseMemObject(dif_d);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(map_kernel);
    clReleaseKernel(reduce_kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    top_nodes(page_ranks, n);

    printf("Time taken :%f for parallel implementation [OpenCL] with %d nodes.\n", float(end - start)/CLOCKS_PER_SEC, n);

    return 0;
}