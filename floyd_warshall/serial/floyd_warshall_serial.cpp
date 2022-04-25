#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cstring>
#include <string>


using namespace std;
void printResults(int op_arr[], int N);
void floydWarshallAlgo(int *mat, const size_t N);


void printResults(int op_arr[], int N)
{
    int INFINITY = 99999;
    ofstream seqfile;
    string output_file_name = "./output/output_serial_" + to_string(N) +".txt";
    seqfile.open(output_file_name);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (op_arr[i*N + j] == INFINITY)
                printf("%7s", "INFINITY");
            else
                seqfile << op_arr[i*N + j] << " ";
        }
        seqfile << endl;
    }
}



void floydWarshallAlgo(int *mat, const size_t N)
{
	for(int k = 0; k < N; k ++)
		for(int i = 0; i < N; i ++)
			for(int j = 0; j < N; j ++)
			{
				int i0 = i*N + j;
				int i1 = i*N + k;
				int i2 = k*N + j;
				if(mat[i1] != -1 && mat[i2] != -1)
                     { 
			          int sum =  (mat[i1] + mat[i2]);
                          if (mat[i0] == -1 || sum < mat[i0])
 					     mat[i0] = sum;
				}
			}
}

int main(int argc, char** argv) {

    // Fetching DataSize :
    if (argc != 2) {
        cout << "Incorrect number of arguments, retry!" << endl;
        exit(1);
    }
    int DATA_SIZE = atoi(argv[1]);

    int matrixCount = DATA_SIZE * DATA_SIZE;
    int *mat = (int*)malloc(sizeof(int) * matrixCount);

    // Read input data
    string input_file_name = "../input_data/input_mat_" + to_string(DATA_SIZE) +".txt";
    ifstream in(input_file_name);
    int number;
    int k = 0;
    while (in >> number) {
        mat[k++] = number;
	}
	in.close();
    printf("k = %d\n", k);

    clock_t start = clock();
    floydWarshallAlgo(mat, DATA_SIZE);
    clock_t end = clock();
    printResults(mat, DATA_SIZE);

    double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout<<"Time taken to run serial code is :"<<time_taken<<" Seconds"<<endl;

}
