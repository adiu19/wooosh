#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define N 64
#define TPB 32
#define K 4
#define MAX_ITER 10
#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    FILE * fp;
    int l = 0;


    float *h_centroids = (float*)malloc(sizeof(float)*K);
    float *h_datapoints = (float*)malloc(N*sizeof(float));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));
    

    srand(time(0));

    fp = fopen("centroids.txt", "r+");
    while (fscanf(fp, "%f", &h_centroids[l++]) != EOF)
    ;
    for(int c=0;c<K;++c)
	{
		// h_centroids[c]=(float) rand() / (double)RAND_MAX;
		printf("%f\n", h_centroids[c]);
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
    l=0;
    fp = fopen("datapoints.txt", "r+");
    while (fscanf(fp, "%f", &h_datapoints[l++]) != EOF)
    ;
	for(int d = 0; d < N; ++d)
	{
		// h_datapoints[d] = (float) rand() / (double)RAND_MAX;
        // printf("%f\n", h_datapoints[d]);
	}


    // Load the kernel source code into the array source_str
    // FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kmeans.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem d_datapoints = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N * sizeof(float), NULL, &ret);
    cl_mem d_clust_assn = clCreateBuffer(context, CL_MEM_READ_WRITE,
            N * sizeof(int), NULL, &ret);
    cl_mem d_centroids = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            K * sizeof(float), NULL, &ret);
    cl_mem d_clust_sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            K * sizeof(float), NULL, &ret);

    // Copy the lists into their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, d_centroids, CL_TRUE, 0,
            K * sizeof(float), h_centroids, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_datapoints, CL_TRUE, 0, 
            N * sizeof(float), h_datapoints, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_clust_sizes, CL_TRUE, 0, 
            K * sizeof(int), h_clust_sizes, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel[2] = {NULL, NULL};
    kernel[0] = clCreateKernel(program, "kMeansClusterAssignment", &ret);
    kernel[1] = clCreateKernel(program, "kMeansCentroidUpdate", &ret);


    int cur_iter = 1;
    int n = N;
    int k = K;
    ret = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_datapoints);
    ret = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_clust_assn);
    ret = clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&n);
    ret = clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&k);
    clFinish(command_queue);

    ret = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&d_datapoints);
    ret = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&d_clust_assn);
    ret = clSetKernelArg(kernel[1], 4, sizeof(int), (void *)&n);
    ret = clSetKernelArg(kernel[1], 5, sizeof(int), (void *)&k);
    clFinish(command_queue);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = 64; // Process the entire lists
    size_t local_item_size = 32; // Process in groups of 64
    cl_event event;

	while(cur_iter < MAX_ITER)
	{
        
        ret = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_centroids);
        clFinish(command_queue);
        
		//call cluster assignment kernel
        ret = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, 
                &global_item_size, &local_item_size, 0, NULL, &event);
        // clWaitForEvents(1, &event);
		// copy new centroids back to host 
        clFinish(command_queue);

        ret = clEnqueueReadBuffer(command_queue, d_centroids, CL_TRUE, 0, 
            K * sizeof(float), h_centroids, 0, NULL, NULL);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for Kernel 1 is",ret);
            exit(1);
        }
        clFinish(command_queue);

		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %f\n",cur_iter,i,h_centroids[i]);
		}

        float zero_float = 0.0;
        int zero_int = 0;
        cl_int err = clEnqueueFillBuffer(command_queue, d_centroids, &zero_float, sizeof(float), 0, K*sizeof(float), 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            perror("Couldn't fill a buffer object");
            exit(1);
        }
        clFinish(command_queue);
        err = clEnqueueFillBuffer(command_queue, d_clust_sizes, &zero_int, sizeof(int), 0, K*sizeof(int), 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            perror("Couldn't fill a buffer object");
            exit(1);
        }
        clFinish(command_queue);
        

        ret = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&d_centroids);
        ret = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void *)&d_clust_sizes);

		//call centroid update kernel
		ret = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, 
                &global_item_size, &local_item_size, 0, NULL, NULL);
        // clWaitForEvents(1, &event);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for Kernel 2 is",ret);
            exit(1);
        }
        clFinish(command_queue);

		cur_iter+=1;
	}

    

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel[0]);
    ret = clReleaseKernel(kernel[1]);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_datapoints);
    ret = clReleaseMemObject(d_clust_assn);
    ret = clReleaseMemObject(d_centroids);
    ret = clReleaseMemObject(d_clust_sizes);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(h_centroids);
    free(h_datapoints);
    free(h_clust_sizes);
    return 0;
}

