

#include <stdio.h>
#include <time.h>

#define N 6400000
#define TPB 32
#define K 8
#define MAX_ITER 20
#define GPU_0 0


__device__ double distance(double x1, double x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void kMeansClusterAssignment(unsigned long long int *d_datapoints, int *d_clust_assn, double *d_centroids)
{
	//get idx for this datapoint
	const unsigned long long int idx = blockIdx.x*blockDim.x + threadIdx.x;

	
	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	double min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;++c)
	{
		double dist = distance((double)d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


__global__ void kMeansCentroidUpdate(unsigned long long int *d_datapoints, int *d_clust_assn, unsigned long long int *d_centroids, unsigned long long int *d_clust_sizes)
{

	//get idx of thread at grid level
	const unsigned long long int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//bounds check
	if (idx >= N) return;



	atomicAdd(&d_centroids[d_clust_assn[idx]],d_datapoints[idx]);
	atomicAdd(&d_clust_sizes[d_clust_assn[idx]],(unsigned long long int)1);

	__syncthreads();
	


}


int main()
{

	//allocate memory on the device for the data points
	unsigned long long int *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	unsigned long long int *d_centroids = 0;
	//allocate memory on the device for the cluster centroids
	double *d_centroids_double = 0;
	//allocate memory on the device for the cluster sizes
	unsigned long long int *d_clust_sizes=0;
	clock_t start, end; // to meaure the time taken by a specific part of code
	double gpu_time_kernel_1;
	double gpu_time_kernel_2;

	FILE * fp;
    unsigned long long int l = 0;

	cudaMalloc(&d_datapoints, N*sizeof(unsigned long long int));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(unsigned long long int));
	cudaMalloc(&d_centroids_double,K*sizeof(double));
	cudaMalloc(&d_clust_sizes,K*sizeof(unsigned long long int));

	unsigned long long int *h_centroids = (unsigned long long int*)malloc(K*sizeof(unsigned long long int));
	double *h_centroids_double = (double*)malloc(K*sizeof(double));
	unsigned long long int *h_datapoints = (unsigned long long int*)malloc(N*sizeof(unsigned long long int));
	unsigned long long int *h_clust_sizes = (unsigned long long int*)malloc(K*sizeof(unsigned long long int));

	srand(time(0));

	char bufC[64];
	snprintf(bufC, 64, "../data/%d_centroids.txt", K);
	//initialize centroids
	fp = fopen(bufC, "r+");
    while (fscanf(fp, "%llu", &h_centroids[l++]) != EOF)
    ;


	for(int c=0;c<K;++c)
	{
		printf("%llu\n", h_centroids[c]);
		h_clust_sizes[c]=0;
		h_centroids_double[c] = (double)h_centroids[c];
	}

	char bufD[64];
	snprintf(bufD, 64, "../data/%d_datapoint.txt", N);
	//initalize datapoints
	l=0;
    fp = fopen(bufD, "r+");
    while (fscanf(fp, "%llu", &h_datapoints[l++]) != EOF)
    ;

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(unsigned long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids_double,h_centroids_double,K*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(unsigned long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(unsigned long long int),cudaMemcpyHostToDevice);

	int cur_iter = 1;
	while(cur_iter < MAX_ITER)
	{	
		start = clock();
		//call cluster assignment kernel
		kMeansClusterAssignment<<<N/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids_double);
		cudaDeviceSynchronize();

		end = clock();
		gpu_time_kernel_1 += (double)(end - start) / CLOCKS_PER_SEC;
		//copy new centroids back to host 
		cudaMemcpy(h_centroids_double,d_centroids_double,K*sizeof(double),cudaMemcpyDeviceToHost);

		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %f\n",cur_iter,i,h_centroids_double[i]);
		}

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids,0,K*sizeof(unsigned long long int));
		cudaMemset(d_clust_sizes,0,K*sizeof(unsigned long long int));
		

		start = clock();
		//call centroid update kernel
		kMeansCentroidUpdate<<<N/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes);
		cudaDeviceSynchronize();

		end = clock();
		gpu_time_kernel_2 += (double)(end - start) / CLOCKS_PER_SEC;


		cudaMemcpy(h_centroids,d_centroids,K*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_sizes,d_clust_sizes,K*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);


		for(int i =0; i < K; ++i){
			h_centroids_double[i] = h_centroids[i]/ (double)h_clust_sizes[i];
		}

		cudaMemcpy(d_centroids_double,h_centroids_double,K*sizeof(double),cudaMemcpyHostToDevice);



		cudaDeviceSynchronize();
		cur_iter+=1;
	}

	printf("Total time taken by the kMeansClusterAssignment kernel is = %lf\n", gpu_time_kernel_1);
	printf("Total time taken by the kMeansCentroidUpdate kernel is = %lf\n", gpu_time_kernel_2);



	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_centroids_double);
	cudaFree(d_clust_sizes);

	free(h_centroids);
	free(h_centroids_double);

	free(h_datapoints);
	free(h_clust_sizes);

	return 0;
}
