#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

__kernel void kMeansClusterAssignment(__global unsigned long *d_datapoints, __global int *d_clust_assn, __global double *d_centroids, const int N, const int K) {
    
    // Get the index of the current element
	unsigned long idx =  get_global_id(0);
    // Do the operation
   if (idx >= N) return;

	//find the closest centroid to this datapoint
	double min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;++c)
	{
		double dist = sqrt((d_centroids[c]-(double)d_datapoints[idx])*(d_centroids[c]-(double)d_datapoints[idx]));

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


__kernel void kMeansCentroidUpdate(__global unsigned long *d_datapoints, __global int *d_clust_assn, __global unsigned long *d_centroids, __global unsigned long *d_clust_sizes, const int N, const int K) {
    
    // Get the index of the current element
   unsigned long idx = get_group_id(1) * get_global_size(0) + get_global_id(0);


	if (idx >= N) return;

	atomic_add(&d_centroids[d_clust_assn[idx]],d_datapoints[idx]);
	atomic_add(&d_clust_sizes[d_clust_assn[idx]],(unsigned long)1);

	barrier(CLK_LOCAL_MEM_FENCE); 

}
