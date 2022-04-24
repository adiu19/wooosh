#include <iostream>
#include <time.h>
#include <random>
#include <math.h>

using namespace std;
int main() {

    unsigned int n = 1000;//256*256;
	unsigned int m = 20000;

    clock_t cpu_start = clock();
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0, 1.0);
	int count = 0;

    printf("starting loop...\n");
	for(int i = 0;i < n;i++){
		//printf("i = %d \n", i);
		int temp = 0;
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
	float pi = 4.0*count/(n*m);
    printf("pi = %f \n", pi);
    printf("time taken on the sequential version is %f \n", (float)(cpu_stop - cpu_start)/CLOCKS_PER_SEC);

}

