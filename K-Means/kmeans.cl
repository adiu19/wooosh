inline float atomicadd(volatile __global float* address, const float value){
    float old = value;
    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
    return old;
}


__kernel void kMeansClusterAssignment(__global float *d_datapoints, __global int *d_clust_assn, __global float *d_centroids, const int N, const int K) {
    
    // Get the index of the current element
   int idx = get_group_id(1) * get_global_size(0) + get_global_id(0);

    // Do the operation
   if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;++c)
	{
		float dist = sqrt((d_centroids[c]-d_datapoints[idx])*(d_centroids[c]-d_datapoints[idx]));

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


__kernel void kMeansCentroidUpdate(__global float *d_datapoints, __global int *d_clust_assn, __global float *d_centroids, __global int *d_clust_sizes, const int N, const int K) {
    
    // Get the index of the current element
   int idx = get_group_id(1) * get_global_size(0) + get_global_id(0);
//    int idx =  get_global_id(0);

	if (idx >= N) return;
    int TPB = 32;
	//get idx of thread at the block level
	const int s_idx = get_local_id(0);

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__local float s_datapoints[32];
	s_datapoints[s_idx]= d_datapoints[idx];

	__local int s_clust_assn[32];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	barrier(CLK_LOCAL_MEM_FENCE); 

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		float b_clust_datapoint_sums[4] = {0,0,0,0};
        // memset( b_clust_datapoint_sums, 0, K*sizeof(int) );
		int b_clust_sizes[4] = {0,0,0,0};
        // memset( b_clust_sizes, 0, K*sizeof(int) );


		for(int j=0; j< get_local_size(0); ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id]+=s_datapoints[j];
			b_clust_sizes[clust_id]+=1;
		}

		// Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
			atomicadd(&d_centroids[z],b_clust_datapoint_sums[z]);
			atomic_add(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	 barrier(CLK_LOCAL_MEM_FENCE); 

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx < K){
		d_centroids[idx] = d_centroids[idx]/d_clust_sizes[idx]; 
	}
}
