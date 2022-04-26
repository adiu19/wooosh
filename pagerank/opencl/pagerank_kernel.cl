__kernel void manage_adj_matrix(__global float* gpu_graph, int n){
    int id = get_global_id(0);
    if(id < n){
        float sum = 0.0;
        for (int i = 0; i < n; ++i){
            sum += gpu_graph[i * n + id];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                gpu_graph[i * n + id] /= sum;
            }
            else{
                gpu_graph[i * n + id] = (1/(float)n);
            }
        }
    }   
}

__kernel void initialize_rank(__global float* gpu_r, int n){
    int id = get_global_id(0);

    if(id < n){
        gpu_r[id] = (1/(float)n);
    }
}

__kernel void store_rank(__global float* gpu_r, __global float* gpu_r_last, int n){
    int id = get_global_id(0);

    if(id < n){
        gpu_r_last[id] = gpu_r[id];
    }
}

__kernel void matmul(__global float* gpu_graph, __global float* gpu_r, __global float* gpu_r_last, int n){
    int id = get_global_id(0);

    if(id < n){
        float sum = 0.0;

        for (int j = 0; j< n; ++j){
            sum += gpu_r_last[j] * gpu_graph[id* n + j];
        }

        gpu_r[id] = sum;
    }
}

__kernel void rank_diff(__global float* gpu_r,__global float* gpu_r_last, int n){
    int id = get_global_id(0);

    if(id < n){
        gpu_r_last[id] = fabs((float)gpu_r_last[id] - (float)gpu_r[id]);
    }
}