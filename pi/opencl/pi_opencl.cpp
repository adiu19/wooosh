#include <iostream>
#include <time.h>
#include <random>
#include <math.h>
#include <CL/cl.h>

using namespace std;

int TPB = 32;
int NBLOCKS = 65536;
typedef numeric_limits<double> DblLim;

#define MAX_SOURCE_SIZE (0x10000000)

int main(){
    clock_t start, end;
	unsigned long n = TPB * NBLOCKS;
	unsigned long m = 10000;
	unsigned long *h_count;
	unsigned long *d_count;

	float pi;

    FILE *fp;
    
    fp = fopen("pi_montecarlo_kernel.cl", "r");
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

    string header_path = "-I mwc64x/cl/mwc64x";
    const char *tmp = header_path.c_str();
    ret = clBuildProgram(program, 1, &device_id, tmp, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        exit(1);
    }

    cl_device_type dev_type;
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
    if (dev_type == CL_DEVICE_TYPE_GPU) {
     //   printf("usage of GPU confirmed \n");
    }

    cl_kernel monte_carlo_pi_kernel = clCreateKernel(program, "monte_carlo_pi_kernel" , &err);

    size_t global_size[1];
    size_t local_size[1];

    global_size[0] = TPB * NBLOCKS;
    local_size[0] = TPB;

    unsigned long zero_int = 0;

    cl_mem g_count;

	h_count = (unsigned long*)malloc(sizeof(unsigned long));

    g_count = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned long), NULL, &err);
    err = clEnqueueFillBuffer(command_queue, g_count, &zero_int, sizeof(unsigned long), 0, sizeof(unsigned long), 0, NULL, NULL);

    err = clSetKernelArg(monte_carlo_pi_kernel, 0, sizeof(cl_mem), &g_count);
    err |= clSetKernelArg(monte_carlo_pi_kernel, 1, sizeof(unsigned long), &n);
    err |= clSetKernelArg(monte_carlo_pi_kernel, 2, sizeof(unsigned long), &m);

    start = clock();
    err = clEnqueueNDRangeKernel(command_queue, monte_carlo_pi_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(command_queue);
    err = clEnqueueReadBuffer(command_queue, g_count, CL_TRUE, 0, sizeof(unsigned long), h_count, 0, NULL, NULL);

    end = clock();
    unsigned long long tests = NBLOCKS * m * TPB;
	cout << "[OpenCL ]Approximated PI using " << tests << " random tests\n";

	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)*h_count/(double)tests << endl;
    cout << "Kernel Execution took " << (double)(end - start)/CLOCKS_PER_SEC << endl;

    clFlush(command_queue);
    clReleaseMemObject(g_count);

    clReleaseProgram(program);

    clReleaseKernel(monte_carlo_pi_kernel);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(h_count);
   
}

