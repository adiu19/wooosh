#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
__global__  void kernel(int *arrDevice, int j, int k);


int main(int argc, char** argv)
{
	if (argc != 2) {
        cout << "Not the correct number of arguments, retry!" << endl;
        exit(1);
    }
    int input = atoi(argv[1]);
    int NUM_VALS = input*input;
	double global_time = 0.0;

	int *arrHost = (int*) malloc(NUM_VALS * sizeof(int));	
	
	std::string input_file_name = "../input_data/input_array_" + std::to_string(input) +".txt";
    ifstream in(input_file_name);
    int number;
    int k = 0;
    while (in >> number) {
        arrHost[k++] = number;
	}
	in.close();
    printf(" k = %d\n", k);
	clock_t start = clock();

	// cuda code starting :
	// for end to end time reading
	int THREADS = 1024;
	int BLOCKS = ceil(NUM_VALS/THREADS);

	//printf("threads : %d , blocks : %d and their product : %d", THREADS, BLOCKS, THREADS*BLOCKS);

    int *arrDevice;
	size_t size = NUM_VALS * sizeof(int);

	cudaMalloc((void**) &arrDevice, size);	
	cudaMemcpy(arrDevice, arrHost, size, cudaMemcpyHostToDevice);

	//clock_t start = clock();
	for (int k = 2; k <= NUM_VALS; k <<= 1)
	{
		for (int j = k >> 1; j > 0; j >>= 1) 
		{
			kernel <<<BLOCKS, THREADS>>>(arrDevice, j, k);
			cudaDeviceSynchronize();
		}
	}
	//global_time += ((double) (clock() - start)); 
	cudaMemcpy(arrHost, arrDevice, size, cudaMemcpyDeviceToHost);
	cudaFree(arrDevice);

  
	//cout << "\nTime : " << ((double) (clock() - start)) / CLOCKS_PER_SEC << " seconds." << endl;
	global_time += ((double) (clock() - start) / CLOCKS_PER_SEC); 
	cout << "\nTime : " << global_time << " seconds." << endl;

    for (int i = 0; i < NUM_VALS - 1; ++i) {
        if (arrHost[i] > arrHost[i + 1]) {
            printf("The order is incorrect :( %d %d at i = %d\n", arrHost[i], arrHost[i + 1], i);
            return 1;
        }
    }
    printf("Successfully Sorted!!\n");
	// global_time += ((double) (clock() - start) / CLOCKS_PER_SEC); 
	// cout << "\nTime : " << global_time << " seconds." << endl;

	//for end to end time reading

}

__global__  void kernel(int *arrDevice, int j, int k)
{
	unsigned int i, pt;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	pt = i^j;

	if ((pt) > i) 
	{
		if ((i&k) == 0) 
		{
			if (arrDevice[i] > arrDevice[pt]) 
			{
				int val = arrDevice[i];
				arrDevice[i] = arrDevice[pt];
				arrDevice[pt] = val;
			}
		}
		if ((i&k) != 0) 
		{
			if (arrDevice[i] < arrDevice[pt]) 
			{
				int val = arrDevice[i];
				arrDevice[i] = arrDevice[pt];
				arrDevice[pt] = val;
			}
		}
	}
}