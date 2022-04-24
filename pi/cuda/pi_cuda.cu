#include <iostream>
#include <time.h>
#include <random>
#include<cuda.h>
#include <curand.h>
#include <math.h>
#include "kernels.cuh"

using namespace std;
int main(){
	unsigned int n = 256*256;
	unsigned int m = 20000;
	int *h_count;
	int *d_count;
	curandState *d_state;
	float pi;


	// allocate memory
	h_count = (int*)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_count, n*sizeof(int));
	cudaMalloc((void**)&d_state, n*sizeof(curandState));
	cudaMemset(d_count, 0, sizeof(int));


	// set up timing stuff
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	setup_kernel<<< gridSize, blockSize>>>(d_state);

	monti_carlo_pi_kernel<<<gridSize, blockSize>>>(d_state, d_count, m);

	cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	pi = *h_count*4.0/(n*m);
	cout<<"Approximate pi calculated on GPU is: "<<pi<<" and calculation took "<<gpu_elapsed_time<<std::endl;

	free(h_count);
	cudaFree(d_count);
	cudaFree(d_state);
}

