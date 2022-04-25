#include "mwc64x_rng.cl"
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void monte_carlo_pi_kernel(__global ulong *g_count, ulong n, ulong m) {
    ulong index = get_global_id(0);
    if (index < n) {

        __local ulong cache[256];
        cache[get_local_id(0)] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        mwc64x_state_t rng;
        MWC64X_SeedStreams(&rng, 12345L, 2*m);
        for(uint i=0;i < m;i++){
            ulong x = MWC64X_NextUint(&rng);
            ulong y = MWC64X_NextUint(&rng);
            ulong x2 = x * x;
            ulong y2 = y * y;
            if(x2 + y2 >= x2) {
                cache[get_local_id(0)]++;
            }    
        }

        ulong i = get_local_size(0)/2;
        while(i != 0){
            if(get_local_id(0) < i){
                cache[get_local_id(0)] += cache[get_local_id(0) + i];
            }

            i /= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(get_local_id(0) == 0){
            atom_add(g_count, cache[0]);
        }
    }
}



