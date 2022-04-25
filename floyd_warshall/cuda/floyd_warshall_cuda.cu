#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define THREAD_SIZE 32
#define INFI 99999

__global__ void floydWarshallKernel(int *matrix, int size, int k);
void printResults(int op_arr[], int N);


void printResults(int op_arr[], int N)
{
    ofstream cudafile;
    string output_file_name = "./output/output_cuda_" + to_string(N) +".txt";
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



__global__ void floydWarshallKernel(int *input, int N, int k)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int a = i * N + j;
    int b = i * N + k;
    int c = k * N + j;

    int pair_i_j = input[a];
    int pair_i_k = input[b];
    int pair_k_j = input[c];

    if(pair_i_k != -1 && pair_k_j != -1)
    {
        int res = pair_i_k + pair_k_j;
        if (pair_i_j == -1 || res < pair_i_j)
            input[a] = res;
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

    // Read input data
    string input_file_name = "../input_data/input_mat_" + to_string(N) +".txt";
    ifstream in(input_file_name);

    int number;
    int k = 0;
    while (in >> number) {
        matHost[k++] = number;
	}
	in.close();

    int *matDevice;
    cudaMalloc((void **)&matDevice, sizeof(int)*N*N);
    cudaMemcpy(matDevice, matHost, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    dim3 dimGrid(N / THREAD_SIZE, N / THREAD_SIZE, 1);
    dim3 dimBlock(THREAD_SIZE, THREAD_SIZE, 1);

    clock_t start = clock();
    // run kernel on GPU 
    for(int k = 0; k < N; ++k) {
        floydWarshallKernel<<<dimGrid, dimBlock>>>(matDevice, N, k);
        cudaDeviceSynchronize();
    }    
    
    global_time += ((double) (clock() - start)) / CLOCKS_PER_SEC; 

    cudaMemcpy(matHost, matDevice, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    cudaFree(matDevice);

    // double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    // cout<<"Time taken to run parallel code on cuda is :"<<time_taken<<" Seconds"<<endl;
    cout << "\nTime : " << global_time << " seconds." << endl;

    printResults(matHost, N);

}