// Approximation of Pi using a simple, and not optimized, CUDA program
#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>

using std::cout;
using std::endl;

typedef unsigned long long Count;
typedef std::numeric_limits<double> DblLim;

const Count TPB = 32;
const Count NBLOCKS = 640;
const Count ITERATIONS = 1000000;

__global__ void picount(Count *totals) {

	__shared__ Count counter[TPB];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	counter[threadIdx.x] = 0;

	for (int i = 0; i < ITERATIONS; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		counter[threadIdx.x] += 1 - int(x * x + y * y);
	}

	if (threadIdx.x == 0) {
		totals[blockIdx.x] = 0;
		for (int i = 0; i < TPB; i++) {
			totals[blockIdx.x] += counter[i];
		}
	}
}

int main(int argc, char **argv) {
	Count *hOut, *dOut;
	hOut = new Count[NBLOCKS];
	cudaMalloc(&dOut, sizeof(Count) * NBLOCKS);

	picount<<<NBLOCKS, TPB>>>(dOut);

	cudaMemcpy(hOut, dOut, sizeof(Count) * NBLOCKS, cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	Count total = 0;
	for (int i = 0; i < NBLOCKS; i++) {
		total += hOut[i];
	}
	Count tests = NBLOCKS * ITERATIONS * TPB;
	cout << "Approximated PI using " << tests << " random tests\n";

	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)total/(double)tests << endl;

	return 0;
}