#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
 #include <math.h>


#define MAX_SOURCE_SIZE (0x100000)
using namespace std;

int main(int argc, char** argv)
{

	if (argc != 2) {
        cout << "Not the correct number of arguments, retry!" << endl;
        exit(1);
    }
    int input = atoi(argv[1]);
    uint32_t arr_size = input*input;
	uint32_t THREAD_SIZE = 1024;

	size_t source_size;
	int *arr = (int*) malloc(arr_size * sizeof(int));	
	char *source = new char[MAX_SOURCE_SIZE];
	
	std::string input_file_name = "../input_data/input_array_" + std::to_string(input) +".txt";
    ifstream in(input_file_name);

	double global_time = 0.0;
    int number;
    int k = 0;
    while (in >> number) {
        arr[k++] = number;
	}
	in.close();
    printf(" k = %d\n", k);

	cl_int err, t_err;
	cl_uint num_devices;
	cl_uint num_platforms;
	cl_context context;
	cl_program program;
	cl_command_queue queue;
	cl_device_id device_id;
	cl_platform_id platform_id;

	// end to end timer clock
    clock_t start = clock();

	err = clGetPlatformIDs(1, &platform_id, &num_platforms);
	err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &t_err);
	err |= t_err;
	queue = clCreateCommandQueue(context, device_id, 0, &t_err);
	err |= t_err;


	FILE *fp = fopen("bitonic_kernel.cl", "rb");
	source_size = fread(source, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_size, &t_err);
	err |= t_err;
	err |= clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);


    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    }

	cl_mem d_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*arr_size, NULL, &t_err);
	err = t_err;
	err = clEnqueueWriteBuffer(queue, d_arr, CL_TRUE, 0, sizeof(int) * arr_size, arr, 0, NULL, NULL);


	cl_kernel kernel = clCreateKernel(program, "sortKernel", &t_err);
	err = t_err;
	err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_arr);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &arr_size);


	size_t global_size[1] = {arr_size};
	size_t local_size[1] = {THREAD_SIZE};

	// start timer here
	// bitonic sort
    //clock_t start = clock();
	for(int i=2; i<=arr_size; i*=2) {
		for(int j=i/2; j>=1; j/=2)
			{
				err = clSetKernelArg(kernel, 1, sizeof(int), &j);
				err |= clSetKernelArg(kernel, 2, sizeof(int), &i);
				err |= clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, local_size, 0, NULL, NULL);
				err |= clFinish(queue);
			}
	}
	//global_time += ((double) (clock() - start)) / CLOCKS_PER_SEC; 
	
	//printf("error code after nested loop is %d \n", err);
	//end timer here

	err |= clEnqueueReadBuffer(queue, d_arr, CL_TRUE, 0, sizeof(int)*arr_size, arr, 0, NULL, NULL);
	err |= clFinish(queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(d_arr);
	err |= clReleaseCommandQueue(queue);
	err |= clReleaseContext(context);
	err |= clReleaseDevice(device_id);


	if(err != CL_SUCCESS)
		cout << "Failure" << endl;
	else
	{
		int i = 1;
		for(; i<arr_size; i++) {
			if(arr[i] < arr[i-1]) {
				cout << "Failure from else " << i << endl;
				break;
			} 	
		}
		if(i==arr_size)
			cout << "Successfully Sorted!!\n" << endl;
	}

	free(source);
	delete[] arr;
	global_time += ((double) (clock() - start) / CLOCKS_PER_SEC); 
	cout << "\nTime : " << global_time << " seconds." << endl;


	return 0;
}