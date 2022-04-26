#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N 640000
#define TPB 32
#define K 8
#define MAX_ITER 20

double distance(double x1, double x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

void kMeansClusterAssignment(unsigned long long int *d_datapoints, int *d_clust_assn, double *d_centroids)
{


	//find the closest centroid to this datapoint

	for(unsigned long long int j = 0 ; j<N ; j++){
		double min_dist = INFINITY;
		int closest_centroid = 0;
		for(int c = 0; c<K;++c)
			{
				double dist = distance((double)d_datapoints[j],d_centroids[c]);

				if(dist < min_dist)
				{
					min_dist = dist;
					closest_centroid=c;
				}
			}
		d_clust_assn[j]=closest_centroid;
		
	}


	//assign closest cluster id for this datapoint/thread
}


void kMeansCentroidUpdate(unsigned long long int *d_datapoints, int *d_clust_assn, double *d_centroids, unsigned long long int *d_clust_sizes)
{

		unsigned long long int b_clust_datapoint_sums[K]={0};
		unsigned long long int b_clust_sizes[K]={0};

		for(unsigned long long int j = 0 ; j<N ; j++){

			b_clust_datapoint_sums[d_clust_assn[j]] += d_datapoints[j];
			b_clust_sizes[d_clust_assn[j]]++;
		
		}

		for(int z=0; z < K; ++z)
		{
			d_centroids[z] = b_clust_datapoint_sums[z]/(double)b_clust_sizes[z];
		}

}


int main()
{


	FILE * fp;
    unsigned long long int l = 0;


	double *h_centroids = (double*)malloc(K*sizeof(double));
	unsigned long long int *h_datapoints = (unsigned long long int*)malloc(N*sizeof(unsigned long long int));
	unsigned long long int *h_clust_sizes = (unsigned long long int*)malloc(K*sizeof(unsigned long long int));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));


	clock_t start, end; // to meaure the time taken by a specific part of code
	double cpu_time_method_1;
	double cpu_time_method_2;

	srand(time(0));

	//initialize centroids
	char bufC[64];
	snprintf(bufC, 64, "../data/%d_centroids.txt", K);
	fp = fopen(bufC, "r+");
    while (fscanf(fp, "%lf", &h_centroids[l++]) != EOF)
    ;


	for(int c=0;c<K;++c)
	{
		printf("%f\n", h_centroids[c]);
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
	char bufD[64];
	snprintf(bufD, 64, "../data/%d_datapoint.txt", N);
	l=0;
    fp = fopen(bufD, "r+");
    while (fscanf(fp, "%llu", &h_datapoints[l++]) != EOF)
    ;


	for(unsigned long long int d = 0; d < N; ++d)
	{
		h_clust_assn[d] = 0;

	}

	int cur_iter = 1;
	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment method
		start = clock();
		kMeansClusterAssignment(h_datapoints,h_clust_assn,h_centroids);

		end = clock();
		cpu_time_method_1 += (double)(end - start) / CLOCKS_PER_SEC;

		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %f\n",cur_iter,i,h_centroids[i]);
		}

		//reset centroids and cluster sizes (will be updated in the next method)
		memset(h_centroids,0.0,K*sizeof(double));
		memset(h_clust_sizes,0,K*sizeof(unsigned long long int));

		//call centroid update method
		start = clock();
		kMeansCentroidUpdate(h_datapoints,h_clust_assn,h_centroids,h_clust_sizes);

		end = clock();
		cpu_time_method_2 += (double)(end - start) / CLOCKS_PER_SEC;

		cur_iter+=1;
	}
	// printf("Total time taken by the kMeansClusterAssignment method is = %lf\n", cpu_time_method_1);
	// printf("Total time taken by the kMeansCentroidUpdate method is = %lf\n", cpu_time_method_2);



	free(h_centroids);
	free(h_datapoints);
	free(h_clust_sizes);

	return 0;
}