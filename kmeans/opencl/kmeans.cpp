#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nvml.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define N 640000
#define TPB 32
#define K 8
#define MAX_ITER 20
#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    FILE * fp;
    unsigned long l = 0;
    clock_t start, end; // to meaure the time taken by a specific part of code
	double gpu_time_kernel_1;
	double gpu_time_kernel_2;


    double *h_centroids_double = (double*)malloc(sizeof(double)*K);
    unsigned long *h_centroids = (unsigned long*)malloc(sizeof(unsigned long)*K);
    unsigned long *h_datapoints = (unsigned long*)malloc(N*sizeof(unsigned long));
	unsigned long *h_clust_sizes = (unsigned long*)malloc(K*sizeof(unsigned long));
    

    srand(time(0));

    char bufC[64];
	snprintf(bufC, 64, "../data/%d_centroids.txt", K);
    fp = fopen(bufC, "r+");
    while (fscanf(fp, "%d", &h_centroids[l++]) != EOF)
    ;
    for(int c=0;c<K;++c)
	{
		printf("%lu\n", h_centroids[c]);
		h_clust_sizes[c]=0;
        h_centroids_double[c] = (double)h_centroids[c];
	}

	//initalize datapoints
    char bufD[64];
	snprintf(bufD, 64, "../data/%d_datapoint.txt", N);
    l=0;
    fp = fopen(bufD, "r+");
    while (fscanf(fp, "%lu", &h_datapoints[l++]) != EOF)
    ;

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
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem d_datapoints = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N * sizeof(unsigned long), NULL, &ret);
    cl_mem d_clust_assn = clCreateBuffer(context, CL_MEM_READ_WRITE,
            N * sizeof(int), NULL, &ret);
    cl_mem d_centroids_double = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            K * sizeof(double), NULL, &ret);
    cl_mem d_centroids = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            K * sizeof(unsigned long), NULL, &ret);
    cl_mem d_clust_sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            K * sizeof(unsigned long), NULL, &ret);

    // Copy the lists into their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, d_centroids, CL_TRUE, 0,
            K * sizeof(unsigned long), h_centroids, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_datapoints, CL_TRUE, 0, 
            N * sizeof(unsigned long), h_datapoints, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_clust_sizes, CL_TRUE, 0, 
            K * sizeof(unsigned long), h_clust_sizes, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_centroids_double, CL_TRUE, 0, 
            K * sizeof(double), h_centroids_double, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);
    }   


    if(ret != CL_SUCCESS) {
        printf("%s %d","Error for clBuildProgram  is",ret);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel[2] = {NULL, NULL};
    kernel[0] = clCreateKernel(program, "kMeansClusterAssignment", &ret);
    if(ret != CL_SUCCESS) {
        printf("%s %d","Error for createKernel 1 is",ret);
        exit(1);
    }
    kernel[1] = clCreateKernel(program, "kMeansCentroidUpdate", &ret);
    if(ret != CL_SUCCESS) {
        printf("%s %d","Error for createKernel 2 is",ret);
        exit(1);
    }


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
    size_t global_item_size = N; // Process the entire lists
    size_t local_item_size = TPB; // Process in groups of 32
    // cl_event event;
    // cl_ulong time_start;
    // cl_ulong time_end;
    // double nanoSeconds = 0;

	while(cur_iter < MAX_ITER)
	{
        
        ret = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_centroids_double);
        clFinish(command_queue);
        
		//call cluster assignment kernel
        start = clock();
        ret = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, 
                &global_item_size, &local_item_size, 0, NULL, NULL);
		// copy new centroids back to host 
        clFinish(command_queue);

        end = clock();
		gpu_time_kernel_1 += (double)(end - start) / CLOCKS_PER_SEC;

        ret = clEnqueueReadBuffer(command_queue, d_centroids_double, CL_TRUE, 0, 
            K * sizeof(double), h_centroids_double, 0, NULL, NULL);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for Kernel 1 is",ret);
            exit(1);
        }
        clFinish(command_queue);

		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %f\n",cur_iter,i,h_centroids_double[i]);
		}

        unsigned long zero_int = 0;
        cl_int err = clEnqueueFillBuffer(command_queue, d_centroids, &zero_int, sizeof(unsigned long), 0, K*sizeof(unsigned long), 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            perror("Couldn't fill a buffer object");
            exit(1);
        }
        clFinish(command_queue);
        err = clEnqueueFillBuffer(command_queue, d_clust_sizes, &zero_int, sizeof(unsigned long), 0, K*sizeof(unsigned long), 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            perror("Couldn't fill a buffer object");
            exit(1);
        }
        clFinish(command_queue);



        // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        // nanoSeconds += time_end-time_start;
        

        ret = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&d_centroids);
        ret = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void *)&d_clust_sizes);

		//call centroid update kernel
        start = clock();
		ret = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, 
                &global_item_size, &local_item_size, 0, NULL, NULL);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for Kernel 2 is",ret);
            exit(1);
        }
        clFinish(command_queue);

        end = clock();
		gpu_time_kernel_2 += (double)(end - start) / CLOCKS_PER_SEC;

        ret = clEnqueueReadBuffer(command_queue, d_centroids, CL_TRUE, 0, 
            K * sizeof(unsigned long), h_centroids, 0, NULL, NULL);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for read buffer is",ret);
            exit(1);
        }
        clFinish(command_queue);

        ret = clEnqueueReadBuffer(command_queue, d_clust_sizes, CL_TRUE, 0, 
            K * sizeof(unsigned long), h_clust_sizes, 0, NULL, NULL);
        if(ret != CL_SUCCESS) {
            printf("%s %d","Error for read buffer is",ret);
            exit(1);
        }
        clFinish(command_queue);

        for(int i =0; i < K; ++i){
			h_centroids_double[i] = h_centroids[i]/ (double)h_clust_sizes[i];
		}


        ret = clEnqueueWriteBuffer(command_queue, d_centroids_double, CL_TRUE, 0,
            K * sizeof(double), h_centroids_double, 0, NULL, NULL);

        clFinish(command_queue);

        

		cur_iter+=1;
	}



    // printf("OpenCl Execution time is: %0.3f milliseconds \n",nanoSeconds / 1000000.0);
    printf("Total time taken by the kMeansClusterAssignment kernel is = %lf\n", gpu_time_kernel_1);
	printf("Total time taken by the kMeansCentroidUpdate kernel is = %lf\n", gpu_time_kernel_2);

    
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
    // printf("%s\n","Hree");

    free(h_centroids);
    free(h_datapoints);
    free(h_clust_sizes);
    return 0;
}

