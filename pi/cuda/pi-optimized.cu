// Approximation of Pi using a simple, and not optimized, CUDA program
#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

typedef unsigned long long Count;
typedef numeric_limits<double> DblLim;

const Count TPB = 32;
const Count NBLOCKS = 65536;
const Count m = 1000000;

__global__ void picount(Count *totals) {

	__shared__ Count counter[TPB];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	counter[threadIdx.x] = 0;

	for (int i = 0; i < m; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		counter[threadIdx.x] += 1 - int(x * x + y * y);
	}

    int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			counter[threadIdx.x] += counter[threadIdx.x + i];
		}

		i /= 2;
		__syncthreads();
	}

	if (threadIdx.x == 0) {
        atomicAdd(totals, counter[0]);
	}
}

int main(int argc, char **argv) {
	Count *hOut, *dOut;
	hOut = new Count[1];
	cudaMalloc(&dOut, sizeof(Count) * 1);

	picount<<<NBLOCKS, TPB>>>(dOut);

	cudaMemcpy(hOut, dOut, sizeof(Count) * 1, cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	Count total = hOut[0];

	Count tests = NBLOCKS * m * TPB;
	cout << "Approximated PI using " << tests << " random tests\n";

	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)total/(double)tests << endl;

	return 0;
}