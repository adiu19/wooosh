#include <iostream>
#include <time.h>
#include <random>
#include <math.h>

const unsigned long long TPB = 1;
const unsigned long long NBLOCKS = 524288;
const unsigned long long m = 10000;
using namespace std;

typedef numeric_limits<double> DblLim;

int main() {

    clock_t cpu_start = clock();
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0, 1.0);
	unsigned long long count = 0;

	unsigned long long n = TPB * NBLOCKS;

	for(unsigned long long i = 0;i < n;i++){
		unsigned long long temp = 0;
		while(temp < m){
			float x = distribution(generator);
			float y = distribution(generator);
			float r = x*x + y*y;

			if(r <= 1){
				count++;
			}
			temp++; 
		}
	}

    clock_t cpu_stop = clock();

	unsigned long long tests = NBLOCKS * m * TPB;
	cout << "[SEQ] approximated PI using " << tests << " random tests\n";

	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)count/(double)tests << endl;

}