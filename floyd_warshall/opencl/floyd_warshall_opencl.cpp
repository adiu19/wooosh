#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <CL/cl.h>

#define INFI 99999
#define MAX_SOURCE_SIZE (0x100000)
using namespace std;

void printResults(int op_arr[], int N);


void printResults(int op_arr[], int N)
{
    ofstream cudafile;
    string output_file_name = "./output/output_cl_" + to_string(N) +".txt";
    cudafile.open(output_file_name);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (op_arr[i*N + j] == INFI)
                printf("%7s", "INFI");
            else
                cudafile << op_arr[i*N + j] << " ";
        }
        cudafile << endl;
    }
}


int main(int argc, char **argv)
{
    
     // Fetching DataSize :
    if (argc != 2) {
        cout << "Incorrect number of arguments, retry!" << endl;
        exit(1);
    }
    int N = atoi(argv[1]);
    double global_time = 0.00;
    int matrixCount = N * N;
    int *matHost = (int*)malloc(sizeof(int) * matrixCount);
    int *path_mat = (int*)malloc(sizeof(int) * matrixCount);

    // Read input data
    string input_file_name = "../input_data/input_mat_" + to_string(N) +".txt";
    ifstream in(input_file_name);

    int number;
    int k = 0;
    while (in >> number) {
        matHost[k++] = number;
    }
    in.close();
    printf( "k = %d ", k);

    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_uint num_of_platforms=0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices=0;
    cl_mem mat_device, path_buffer;
    
    if(clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
    {
        printf("Not fetching platform id\n");
        return 1;
    }


    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_of_devices) != CL_SUCCESS)
    {
    printf("Not fetching device_id\n");
    return 1;
    }
    //printf("platform id : %d and device id : %d", platform_id, device_id);

    properties[0]= CL_CONTEXT_PLATFORM;
    properties[1]= (cl_context_properties) platform_id;
    properties[2]= 0;

    context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	
    size_t source_size;
	char *source = new char[MAX_SOURCE_SIZE];

    FILE *fp = fopen("kernel.cl", "rb");
	source_size = fread(source, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_size, &err);

    //program = clCreateProgramWithSource(context, 1, (const char **) &ProgramSource, NULL, &err);

    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error building program\n");
        return 1;
    }

    kernel = clCreateKernel(program, "fw_kernel", &err);

    mat_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * matrixCount, NULL, NULL);
    path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * matrixCount, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, mat_device, CL_TRUE, 0, sizeof(int) * matrixCount, matHost, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, path_buffer, CL_TRUE, 0, sizeof(int) * matrixCount, path_mat, 0, NULL, NULL);


    int temp = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_device);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &path_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &temp);
    clSetKernelArg(kernel, 3, sizeof(int), &temp);

    size_t global[2];
    size_t local[2];
    int BLOCK_SIZE = 32;

    global[0] = N;
    global[1] = N;
    local[0] = BLOCK_SIZE;
    local[1] = BLOCK_SIZE;
    int num_passes = N;

    clock_t start = clock();

    for(int i=0; i<num_passes; i++)
    {
        clSetKernelArg(kernel, 3, sizeof(int), &i);
        clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clFlush(command_queue);
    }

    global_time += ((double) (clock() - start)) / CLOCKS_PER_SEC; 

    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, mat_device, CL_TRUE, 0, sizeof(int) * matrixCount, matHost, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, path_buffer, CL_TRUE, 0, sizeof(int) * matrixCount, path_mat, 0, NULL, NULL);

    printResults(matHost, N);

    //double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    //cout<<"Time taken to run the parallel code on opencl is :"<<time_taken<<" Seconds"<<endl;
    cout << "\nTime : " << global_time << " seconds." << endl;

    clReleaseMemObject(mat_device);
    clReleaseMemObject(path_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    
    return 0;
}